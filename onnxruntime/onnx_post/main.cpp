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
	
	auto output_dims = ort_outputs[3].GetTensorTypeAndShapeInfo().GetShape();
	num_proposal = output_dims[0];
	nout = output_dims[1];
																																						 /////generate proposals
	vector<BoxInfo> generate_boxes;
	float ratioh = (float)frame.rows / this->inpHeight, ratiow = (float)frame.cols / this->inpWidth;
	int n = 0, k = 0;
	const int64* pbox = ort_outputs[3].GetTensorMutableData<int64>();
	const float* pscore = ort_outputs[2].GetTensorMutableData<float>();
	for (n = 0; n < this->num_proposal; n++)   ///特征图尺度
	{
		float score = pscore[n];
		if (score > this->confThreshold)
		{
			int classid = (int)pbox[1];
			float ymin = (float)pbox[2] * ratioh;  
			float xmin = (float)pbox[3] * ratiow;
			float ymax = (float)pbox[4] * ratioh;
			float xmax = (float)pbox[5] * ratiow;
			generate_boxes.push_back(BoxInfo{ xmin, ymin, xmax, ymax, score, classid });
		}
		pbox += nout;
		pscore++;
	}

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
	int i = 0, j = 0, area = this->inpHeight*this->inpWidth;
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
	Net_config YOLOPv2_nets = { 0.5, 0.5, "onnx_post/yolopv2_post_192x320.onnx" };   ////choices = onnx_post文件夹里的文件
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