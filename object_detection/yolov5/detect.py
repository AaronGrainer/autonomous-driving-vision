import torch

from detectron2.utils.visualizer import Visualizer

import cv2
import numpy as np
import pandas as pd
import fire
import matplotlib
import matplotlib.pyplot as plt
import time

from object_detection.utils.sort import Sort


class YoloDetector:
    def __init__(self):
        self.max_age = 1
        self.min_hits = 3
        self.iou_threshold = 0.3

        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.mot_tracker = Sort(
            max_age=self.max_age,
            min_hits=self.min_hits,
            iou_threshold=self.iou_threshold
        )
        self.classes = None
        matplotlib.use('TkAgg')

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def _predict(self, img):
        with torch.no_grad():
            results = self.model(img)
            if not self.classes:
                self.classes = results.names
            preds = results.pandas().xyxy[0]

        return preds

    def _update_tracker(self, preds):
        class_unique = {key: 1 for key in self.classes}

        detections = [[pred['xmin'], pred['ymin'], pred['xmax'], pred['ymax'], pred['confidence'], pred['class']]
                      for _, pred in preds.iterrows()]
        
        trackers = self.mot_tracker.update(detections)

        trackers_formatted = list()
        for track in trackers:
            class_name = self.classes[int(track[5])]
            class_unique_id = class_unique[class_name]
            trackers_formatted.append(np.append(track, [class_name, class_unique_id]))
            class_unique.update({
                class_name: class_unique_id + 1
            })
        
        tracker_df = pd.DataFrame(trackers_formatted, columns=['xmin', 'ymin', 'xmax', 'ymax', 'track_id', 'class', 'name', 'class_unique_id'])
        tracker_df[['xmin', 'ymin', 'xmax', 'ymax', 'track_id', 'class']] = \
            tracker_df[['xmin', 'ymin', 'xmax', 'ymax', 'track_id', 'class']].apply(pd.to_numeric, downcast='integer')

        return tracker_df

    def _visualize(self, img, preds):
        v = Visualizer(img, scale=1.5)

        for _, pred in preds.iterrows():
            box_coord = (pred['xmin'], pred['ymin'], pred['xmax'], pred['ymax'])
            v.draw_box(box_coord, edge_color='moccasin')
            v.draw_text(f"{pred['name']} {pred['class_unique_id']}", (pred['xmin'], max(0, pred['ymin'] - 14)), font_size=8,
                        color='moccasin', horizontal_alignment='left')
        
        out = v.get_output()
        return out.get_image()[:, :, ::-1]

    def detect(self, img, visualize=True, return_preds=False):
        preds = self._predict(img)
        preds = self._update_tracker(preds)

        if visualize:
            return self._visualize(img, preds)
        elif return_preds:
            return preds
        else:
            return

    def detect_img(self, img):
        img_pred = self.detect(img)
        cv2.imshow('preds', img_pred)
        cv2.waitKey()

    def detect_video(self, input_video):
        cap = cv2.VideoCapture(input_video)
        ret, frame = cap.read()

        ax1 = plt.subplot(111)
        im1 = ax1.imshow(frame[:, :, ::-1])

        plt.ion()
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is True:
                start_time = time.time()
                frame = self.detect(frame[:, :, ::-1])
                im1.set_data(frame[:, :, ::-1])
                plt.pause(0.001)
                print("FPS: ", 1.0 / (time.time() - start_time))
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
        input_video = 'test_asset/usa_laguna.mp4'

        yolo_detector = YoloDetector()
        yolo_detector.detect_video(input_video)
    else:
        raise ValueError


if __name__ == '__main__':
    fire.Fire(main)

