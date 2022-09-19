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

        scores_ = results[2]
        batchno_classid_y1x1y2x2 = results[3]
        # Traffic Object Detection
        bboxes, scores, class_ids = [], [], []
        for score, batchno_classid_y1x1y2x2_ in zip(scores_, batchno_classid_y1x1y2x2):
            if score < self.confThreshold:
                continue

            class_id = int(batchno_classid_y1x1y2x2_[1])
            y1 = batchno_classid_y1x1y2x2_[2]
            x1 = batchno_classid_y1x1y2x2_[3]
            y2 = batchno_classid_y1x1y2x2_[4]
            x2 = batchno_classid_y1x1y2x2_[5]
            y1 = int(y1 * ratioh)
            x1 = int(x1 * ratiow)
            y2 = int(y2 * ratioh)
            x2 = int(x2 * ratiow)

            bboxes.append([x1, y1, x2, y2])
            class_ids.append(class_id)
            scores.append(score)

        for bbox, score, class_id in zip(bboxes, scores, class_ids):
            x1, y1 = int(bbox[0]), int(bbox[1])
            x2, y2 = int(bbox[2]), int(bbox[3])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, '%s:%.2f' % (self.classes[class_id-1], score), (x1, y1 - 5), 0,
                       0.7, (0, 255, 0), 2)

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
    parser.add_argument("--modelpath", type=str, default='onnx_post/yolopv2_post_192x320.onnx', help="model path")
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
