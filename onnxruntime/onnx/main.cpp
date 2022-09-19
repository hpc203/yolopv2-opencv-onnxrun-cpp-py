#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>

using namespace std;
using namespace cv;
using namespace Ort;

struct Net_config
{
	float confThreshold; // Confidence threshold
	float nmsThreshold;  // Non-maximum suppression threshold
	string modelpath;
};

typedef struct BoxInfo
{
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	int label;
} BoxInfo;

class YOLOPv2
{
public:
	YOLOPv2(Net_config config);
	Mat detect(Mat& frame);
private:
	int inpWidth;
	int inpHeight;
	int nout;
	int num_proposal;
	vector<string> class_names;
	int num_class;

	float confThreshold;
	float nmsThreshold;
	vector<float> input_image_;
	void normalize_(Mat img);
	void nms(vector<BoxInfo>& input_boxes);
	const float anchors[3][6] = { {12, 16, 19, 36, 40, 28}, {36, 75, 76, 55, 72, 146},{142, 110, 192, 243, 459, 401} };
	const float stride[3] = { 8.0, 16.0, 32.0 };

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "YOLOPv2");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims; // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
};

YOLOPv2::YOLOPv2(Net_config config)
{
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;

	string model_path = config.modelpath;
	std::wstring widestr = std::wstring(model_path.begin(), model_path.end());
	//OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, widestr.c_str(), sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	this->inpHeight = input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3];

	string classesFile = "coco.names";
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) this->class_names.push_back(line);
	this->num_class = class_names.size();
}

void YOLOPv2::normalize_(Mat img)
{
	//    img.convertTo(img, CV_32F);
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(row * col * img.channels());
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				float pix = img.ptr<uchar>(i)[j * 3 + 2 - c];
				this->input_image_[c * row * col + i * col + j] = pix / 255.0;
			}
		}
	}
}

void YOLOPv2::nms(vector<BoxInfo>& input_boxes)
{
	sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
	vector<float> vArea(input_boxes.size());
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
			* (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
	}

	vector<bool> isSuppressed(input_boxes.size(), false);
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		if (isSuppressed[i]) { continue; }
		for (int j = i + 1; j < int(input_boxes.size()); ++j)
		{
			if (isSuppressed[j]) { continue; }
			float xx1 = (max)(input_boxes[i].x1, input_boxes[j].x1);
			float yy1 = (max)(input_boxes[i].y1, input_boxes[j].y1);
			float xx2 = (min)(input_boxes[i].x2, input_boxes[j].x2);
			float yy2 = (min)(input_boxes[i].y2, input_boxes[j].y2);

			float w = (max)(float(0), xx2 - xx1 + 1);
			float h = (max)(float(0), yy2 - yy1 + 1);
			float inter = w * h;
			float ovr = inter / (vArea[i] + vArea[j] - inter);

			if (ovr >= this->nmsThreshold)
			{
				isSuppressed[j] = true;
			}
		}
	}
	// return post_nms;
	int idx_t = 0;
	input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &isSuppressed](const BoxInfo& f) { return isSuppressed[idx_t++]; }), input_boxes.end());
}

inline float sigmoid(float x)
{
	return 1.0 / (1 + exp(-x));
}

Mat YOLOPv2::detect(Mat& frame)
{
	Mat dstimg;
	resize(frame, dstimg, Size(this->inpWidth, this->inpHeight));
	this->normalize_(dstimg);
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	// 开始推理
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());   // 开始推理
																																				 /////generate proposals
	vector<BoxInfo> generate_boxes;
	float ratioh = (float)frame.rows / this->inpHeight, ratiow = (float)frame.cols / this->inpWidth;
	int n = 0, q = 0, i = 0, j = 0, nout = this->class_names.size() + 5, c = 0, area = 0;
	for (n = 0; n < 3; n++)   ///尺度
	{
		int num_grid_x = (int)(this->inpWidth / this->stride[n]);
		int num_grid_y = (int)(this->inpHeight / this->stride[n]);
		area = num_grid_x * num_grid_y;
		const float* pdata = ort_outputs[n + 2].GetTensorMutableData<float>();
		for (q = 0; q < 3; q++)    ///anchor数
		{
			const float anchor_w = this->anchors[n][q * 2];
			const float anchor_h = this->anchors[n][q * 2 + 1];
			for (i = 0; i < num_grid_y; i++)
			{
				for (j = 0; j < num_grid_x; j++)
				{
					float box_score = sigmoid(pdata[4 * area + i * num_grid_x + j]);
					if (box_score > this->confThreshold)
					{
						float max_class_socre = -100000, class_socre = 0;
						int max_class_id = 0;
						for (c = 0; c < this->class_names.size(); c++) //// get max socre
						{
							class_socre = pdata[(c + 5) * area + i * num_grid_x + j];
							if (class_socre > max_class_socre)
							{
								max_class_socre = class_socre;
								max_class_id = c;
							}
						}
						max_class_socre = sigmoid(max_class_socre) * box_score;
						if (max_class_socre > this->confThreshold)
						{
							float cx = (sigmoid(pdata[i * num_grid_x + j]) * 2.f - 0.5f + j) * this->stride[n];  ///cx
							float cy = (sigmoid(pdata[area + i * num_grid_x + j]) * 2.f - 0.5f + i) * this->stride[n];   ///cy
							float w = powf(sigmoid(pdata[2 * area + i * num_grid_x + j]) * 2.f, 2.f) * anchor_w;   ///w
							float h = powf(sigmoid(pdata[3 * area + i * num_grid_x + j]) * 2.f, 2.f) * anchor_h;  ///h

							float xmin = (cx - 0.5*w)*ratiow;
							float ymin = (cy - 0.5*h)*ratioh;
							float xmax = (cx + 0.5*w)*ratiow;
							float ymax = (cy + 0.5*h)*ratioh;

							generate_boxes.push_back(BoxInfo{ xmin, ymin, xmax, ymax, max_class_socre, max_class_id });
						}
					}
				}
			}
			pdata += area * nout;
		}
	}
	nms(generate_boxes);

	Mat outimg = frame.clone();
	for (size_t i = 0; i < generate_boxes.size(); ++i)
	{
		int xmin = int(generate_boxes[i].x1);
		int ymin = int(generate_boxes[i].y1);
		rectangle(outimg, Point(xmin, ymin), Point(int(generate_boxes[i].x2), int(generate_boxes[i].y2)), Scalar(0, 0, 255), 2);
		string label = format("%.2f", generate_boxes[i].score);
		label = this->class_names[generate_boxes[i].label-1] + ":" + label;
		putText(outimg, label, Point(xmin, ymin - 5), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
	}

	const float* pdrive_area = ort_outputs[0].GetTensorMutableData<float>();
	const float* plane_line = ort_outputs[1].GetTensorMutableData<float>();
	area = this->inpHeight*this->inpWidth;
	for (i = 0; i < frame.rows; i++)
	{
		for (j = 0; j < frame.cols; j++)
		{
			const int x = int(j / ratiow);
			const int y = int(i / ratioh);
			if (pdrive_area[y * this->inpWidth + x] < pdrive_area[area + y * this->inpWidth + x])
			{
				outimg.at<Vec3b>(i, j)[0] = 0;
				outimg.at<Vec3b>(i, j)[1] = 255;
				outimg.at<Vec3b>(i, j)[2] = 0;
			}
			if (plane_line[y * this->inpWidth + x] > 0.5)
			{
				outimg.at<Vec3b>(i, j)[0] = 255;
				outimg.at<Vec3b>(i, j)[1] = 0;
				outimg.at<Vec3b>(i, j)[2] = 0;
			}
		}
	}
	return outimg;
}

int main()
{
	Net_config YOLOPv2_nets = { 0.5, 0.5, "onnx/yolopv2_192x320.onnx" };   ////choices = onnx文件夹里的文件
	YOLOPv2 net(YOLOPv2_nets);
	string imgpath = "images/0ace96c3-48481887.jpg";
	Mat srcimg = imread(imgpath);
	Mat outimg = net.detect(srcimg);

	static const string kWinName = "Deep learning object detection in ONNXRuntime";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, outimg);
	waitKey(0);
	destroyAllWindows();
}