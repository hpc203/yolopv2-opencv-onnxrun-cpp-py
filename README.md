# yolopv2-opencv-onnxrun-cpp-py
分别使用OpenCV、ONNXRuntime部署YOLOPV2目标检测+可驾驶区域分割+车道线分割，一共包含54个onnx模型，依然是包含C++和Python两个版本的程序

由于模型文件数量比较多，无法直接上传到github，因此把模型文件上传到百度云盘。
链接: https://pan.baidu.com/s/1aOflEqusdGZT2mhQI-ckqg  密码: w6g9

一共有7.84G

其中
(1).onnx_post文件夹里的onnx文件，是把最后3个yolo层在经过decode之后，经过torch.cat(z, 1)合并成一个张量，并且还包含nms的。
因此在加载onnx_post文件夹里的onnx文件做推理之后的后处理非常简单，只需要过滤置信度低的检测框。但是经过程序运行实验，onnxruntime能加载
onnx文件做推理并且结果正常，但是opencv的dnn模块不能。

(2). onnx_split文件夹里的onnx文件，是把最后3个yolo层在经过decode之后，经过torch.cat(z, 1)合并成一个张量。
因此在加载onnx_split文件夹里的onnx文件做推理之后的后处理，包括过滤置信度低的检测框，然后执行nms去除重叠度高的检测框。
经过程序运行实验，onnxruntime能加载onnx文件做推理并且结果正常，而opencv的dnn模块能加载onnx文件，但是在forward函数报错

(3). onnx文件夹里的onnx文件, 是不包含最后3个yolo层的。因此在加载onnx_split文件夹里的onnx文件做推理之后的后处理，包括
3个yolo层分别做decode，过滤置信度低的检测框，执行nms去除重叠度高的检测框，一共3个步骤。
经过程序运行实验，onnxruntime和opencv的dnn模块都能加载onnx文件做推理并且结果正常。
