#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

struct Net_config
{
	float confThreshold; // Confidence threshold
	float nmsThreshold;  // Non-maximum suppression threshold
	string modelpath;
};

class YOLOPv2
{
public:
	YOLOPv2(Net_config config);
	Mat detect(Mat& frame);
private:
	int inpWidth;
	int inpHeight;
	vector<string> class_names;
	int num_class;

	float confThreshold;
	float nmsThreshold;
	Net net;
	const float anchors[3][6] = { {12, 16, 19, 36, 40, 28}, {36, 75, 76, 55, 72, 146},{142, 110, 192, 243, 459, 401} };
	const float stride[3] = { 8.0, 16.0, 32.0 };
	void drawPred(float conf, int left, int top, int right, int bottom, Mat& frame, int classid);
};

YOLOPv2::YOLOPv2(Net_config config)
{
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;

	this->net = readNet(config.modelpath);
	ifstream ifs("coco.names");
	string line;
	while (getline(ifs, line)) this->class_names.push_back(line);
	this->num_class = class_names.size();

	size_t pos = config.modelpath.rfind("_");
	size_t pos_ = config.modelpath.rfind(".");
	int len = pos_ - pos - 1;
	string hxw = config.modelpath.substr(pos + 1, len);
	size_t position = hxw.find("Nx3x");
	if (position != hxw.npos)
	{
		len = hxw.length();
		hxw = hxw.substr(position + 4, len - 4);
	}
	pos = hxw.rfind("x");
	string h = hxw.substr(0, pos);
	len = hxw.length() - pos;
	string w = hxw.substr(pos + 1, len);
	this->inpHeight = stoi(h);
	this->inpWidth = stoi(w);
}

void YOLOPv2::drawPred(float conf, int left, int top, int right, int bottom, Mat& frame, int classid)   // Draw the predicted bounding box
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 2);

	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);
	label = this->class_names[classid-1] + ":" + label;

	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	//rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
}

inline float sigmoid(float x)
{
	return 1.0 / (1 + exp(-x));
}

Mat YOLOPv2::detect(Mat& frame)
{
	Mat blob = blobFromImage(frame, 1 / 255.0, Size(this->inpWidth, this->inpHeight), Scalar(0, 0, 0), true, false);
	this->net.setInput(blob);
	vector<Mat> outs;
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());   // 开始推理
																																				 
	vector<float> confidences;
	vector<Rect> boxes;
	vector<int> classIds;
	float ratioh = (float)frame.rows / this->inpHeight, ratiow = (float)frame.cols / this->inpWidth;
	int n = 0, q = 0, i = 0, j = 0, nout = this->class_names.size() + 5, c = 0, area = 0;
	for (n = 0; n < 3; n++)   ///尺度
	{
		int num_grid_x = (int)(this->inpWidth / this->stride[n]);
		int num_grid_y = (int)(this->inpHeight / this->stride[n]);
		area = num_grid_x * num_grid_y;
		float* pdata = (float*)outs[n * 2].data;
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

							int left = int((cx - 0.5*w)*ratiow);
							int top = int((cy - 0.5*h)*ratioh);
							int width = int(w*ratiow);
							int height = int(h*ratioh);

							confidences.push_back(max_class_socre);
							boxes.push_back(Rect(left, top, (int)(width), (int)(height)));
							classIds.push_back(max_class_id);
						}
					}
				}
			}
			pdata += area * nout;
		}
	}
	vector<int> indices;
	dnn::NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);

	Mat outimg = frame.clone();
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		this->drawPred(confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, outimg, classIds[idx]);
	}

	float* pdrive_area = (float*)outs[1].data;
	float* plane_line = (float*)outs[3].data;
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
	Net_config YOLOPv2_nets = { 0.5, 0.5, "onnx/yolopv2_Nx3x384x1280.onnx" };   ////choices = onnx文件夹里的文件
	YOLOPv2 net(YOLOPv2_nets);
	string imgpath = "images/0ace96c3-48481887.jpg";
	Mat srcimg = imread(imgpath);
	Mat outimg = net.detect(srcimg);

	static const string kWinName = "Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, outimg);
	waitKey(0);
	destroyAllWindows();
}