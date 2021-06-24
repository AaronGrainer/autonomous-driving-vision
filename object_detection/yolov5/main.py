import torch

from detectron2.utils.visualizer import Visualizer

import cv2
import numpy as np
import pandas as pd
import fire
import matplotlib
import matplotlib.pyplot as plt


class YoloDetector:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        matplotlib.use('TkAgg')

    def _visualize(self, img, predictions):
        v = Visualizer(img, scale=1)

        for _, pred in predictions.iterrows():
            box_coord = (pred['xmin'], pred['ymin'], pred['xmax'], pred['ymax'])
            v.draw_box(box_coord, edge_color='moccasin')
            v.draw_text(pred['name'], (pred['xmin'], pred['ymin']), font_size=8,
                        color='moccasin', horizontal_alignment='left')
        out = v.get_output()
        return out.get_image()[:, :, ::-1]

    def _detect(self, img):
        results = self.model(img)
        # classes = results.names
        # print('classes: ', classes)
        predictions = results.pandas().xyxy[0]
        return self._visualize(img, predictions)

    def detect_img(self, img):
        img_pred = self._detect(img)
        cv2.imshow('Predictions', img_pred)
        cv2.waitKey()

    def detect_video(self):
        input_video = 'test_asset/usa_laguna.mp4'

        cap = cv2.VideoCapture(input_video)
        ret, frame = cap.read()

        ax1 = plt.subplot(111)
        im1 = ax1.imshow(frame[:, :, ::-1])

        plt.ion()

        while cap.isOpened():
            ret, frame = cap.read()
            if ret is True:
                frame = self._detect(frame[:, :, ::-1])
                im1.set_data(frame[:, :, ::-1])
                plt.pause(0.2)
            else:
                break

        plt.ioff()
        plt.show()


def main(detect_type='image'):
    if detect_type == 'image':
        img_path = 'test_asset/usa_laguna_moment.jpg'
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

