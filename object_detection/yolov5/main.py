import torch

from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances, Boxes

import cv2
import pandas as pd
import fire


class YoloDetector:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    def _visualize(self, img, predictions):
        v = Visualizer(img, scale=1)

        for _, pred in predictions.iterrows():
            box_coord = (pred['xmin'], pred['ymin'], pred['xmax'], pred['ymax'])
            v.draw_box(box_coord, edge_color='moccasin')
        out = v.get_output()
        return out.get_image()[:, :, ::-1]

    def _detect(self, img):
        results = self.model(img)
        # classes = results.names
        predictions = results.pandas().xyxy[0]
        return self._visualize(img, predictions)

    def detect_img(self, img):
        img_pred = self._detect(img, img)
        cv2.imshow('Predictions', img_pred)
        cv2.waitKey()

    def detect_video(self):
        input_video = 'test_asset/usa_laguna.mp4'
        output_video = 'object_detection/yolov5/output/yolov5_output.avi'

        cap = cv2.VideoCapture(input_video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        vout = cv2.VideoWriter(output_video, fourcc, 30.0, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if ret is True:
                frame = self._detect(frame)
                vout.write(frame[:, :, ::-1])
            else:
                break

        vout.release()


def main(detect_type='image'):
    if detect_type == 'image':
        img_path = 'test_asset/solidWhiteCurve.jpg'
        img = cv2.imread(img_path)[:, :, ::-1]

        yolo_detector = YoloDetector()
        yolo_detector.detect_img(img)
    elif detect_type == 'video':
        yolo_detector = YoloDetector()
        yolo_detector.detect_video()
    else:
        raise NotImplemented


if __name__ == '__main__':
    fire.Fire(main)

