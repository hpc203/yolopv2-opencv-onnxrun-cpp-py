#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import cv2
import numpy as np
import onnxruntime as ort

class YOLOPv2():
    def __init__(self, model_path, confThreshold=0.5):
        self.classes = list(map(lambda x: x.strip(), open('coco.names', 'r').readlines()))
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession(model_path, so)
        model_inputs = self.session.get_inputs()
        self.input_name = model_inputs[0].name
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = int(self.input_shape[2])
        self.input_width = int(self.input_shape[3])
        self.confThreshold = confThreshold
        self.nmsThreshold = 0.5

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)

        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId-1], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
        cv2.putText(frame, label, (left, top - 10), 0, 0.7, (0, 255, 0), thickness=2)
        return frame
    def detect(self, frame):
        image_width, image_height = frame.shape[1], frame.shape[0]
        ratioh = image_height / self.input_height
        ratiow = image_width / self.input_width

        # Pre process:Resize, BGR->RGB, float32 cast
        input_image = cv2.resize(frame, dsize=(self.input_width, self.input_height))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, axis=0)
        input_image = input_image.astype('float32')
        input_image = input_image / 255.0

        # Inference
        results = self.session.run(None, {self.input_name: input_image})

        det_out = results[2].squeeze(axis=0)

        boxes, confidences, classIds = [], [], []
        for i in range(det_out.shape[0]):
            # if det_out[i, 4] < self.confThreshold:
            #     continue

            if det_out[i, 4] * np.max(det_out[i, 5:]) < self.confThreshold:
                continue

            class_id = np.argmax(det_out[i, 5:])
            cx, cy, w, h = det_out[i, :4]
            x = int((cx - 0.5*w) * ratiow)
            y = int((cy - 0.5*h) * ratioh)
            width = int(w * ratiow)
            height = int(h* ratioh)

            boxes.append([x, y, width, height])
            classIds.append(class_id)
            confidences.append(det_out[i, 4] * np.max(det_out[i, 5:]))
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            frame = self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)

        # Drivable Area Segmentation
        drivable_area = np.squeeze(results[0], axis=0)
        mask = np.argmax(drivable_area, axis=0).astype(np.uint8)
        mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
        frame[mask==1] = [0, 255, 0]
        # Lane Line
        lane_line = np.squeeze(results[1])
        mask = np.where(lane_line > 0.5, 1, 0).astype(np.uint8)
        mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
        frame[mask==1] = [255, 0, 0]
        return frame

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelpath", type=str, default='onnx_split/yolopv2_split_192x320.onnx', help="model path")
    parser.add_argument("--imgpath", type=str, default='images/0ace96c3-48481887.jpg', help="image path")
    parser.add_argument("--confThreshold", default=0.5, type=float, help='class confidence')
    args = parser.parse_args()

    net = YOLOPv2(args.modelpath, confThreshold=args.confThreshold)
    srcimg = cv2.imread(args.imgpath)
    srcimg = net.detect(srcimg)

    winName = 'Deep learning object detection in ONNXRuntime'
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    cv2.imshow(winName, srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
