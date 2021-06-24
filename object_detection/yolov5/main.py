import torch

from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances, Boxes

import cv2
import pandas as pd


class YoloDetector:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    def _visualize(self, img, predictions):
        v = Visualizer(img, scale=1.5)

        for _, pred in predictions.iterrows():
            box_coord = (pred['xmin'], pred['ymin'], pred['xmax'], pred['ymax'])
            v.draw_box(box_coord, edge_color='moccasin')
        out = v.get_output()
        return out.get_image()[:, :, ::-1]

    def detect_img(self, img):
        results = self.model(img)
        # classes = results.names
        predictions = results.pandas().xyxy[0]
        img_pred = self._visualize(img, predictions)

        cv2.imshow('Predictions', img_pred)
        cv2.waitKey()


def main():
    img_path = 'test_asset/solidWhiteCurve.jpg'
    img = cv2.imread(img_path)[:, :, ::-1]

    yolo_detector = YoloDetector()
    yolo_detector.detect_img(img)


if __name__ == '__main__':
    main()

