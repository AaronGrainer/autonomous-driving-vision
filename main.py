from detectron2.utils.visualizer import Visualizer

from object_detection.yolov5.detect import YoloDetector
from object_detection.utils.trafic_light import detect_trafic_light_color
from depth_estimation.monodepth.detect import MonoDepthEstimator

import cv2
import matplotlib.pyplot as plt
import fire
import time


class AutonomousDetector:
    def __init__(self):
        self.yolo_detector = YoloDetector()
        self.mono_depth_estimator = MonoDepthEstimator()

    def _visualize(self, img, preds):
        v = Visualizer(img, scale=1.5)

        for _, pred in preds.iterrows():
            box_coord = (pred['xmin'], pred['ymin'], pred['xmax'], pred['ymax'])
            v.draw_box(box_coord, edge_color='moccasin')

            classname = pred['name'].title()
            if classname == 'Car':
                text = f"{classname} ({pred['depth']:.1f}m)\n" \
                       f"ID: {pred['class_unique_id']}"
            elif classname == 'Traffic Light':
                traffic_color = f"({pred['traffic_color']})" if pred['traffic_color'] != 'other' else ''
                text = f"{classname} {traffic_color}\n" \
                       f"ID: {pred['class_unique_id']}"
            else:
                text = f"{classname}\n" \
                       f"ID: {pred['class_unique_id']}"

            v.draw_text(text, (pred['xmin'], max(0, pred['ymin'] - 16)), font_size=5,
                        color='moccasin', horizontal_alignment='left')
        
        out = v.get_output()
        return out.get_image()[:, :, ::-1]

    def detect(self, img):
        preds = self.yolo_detector.detect(img, visualize=False, return_preds=True)
        preds = self.mono_depth_estimator.detect_img_object(img, preds, class_only=['car'])
        preds = detect_trafic_light_color(img, preds)
        return self._visualize(img, preds)

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
                plt.pause(0.005)
                print("FPS: ", 1.0 / (time.time() - start_time))
            else:
                break
        plt.ioff()
        plt.show()


def main(detect_type='image'):
    if detect_type == 'image':
        img_path = 'test_asset/usa_laguna_moment.jpg'
        img = cv2.imread(img_path)[:, :, ::-1]

        autonomous_detector = AutonomousDetector()
        autonomous_detector.detect_img(img)
    elif detect_type == 'video':
        input_video = 'test_asset/usa_laguna.mp4'

        autonomous_detector = AutonomousDetector()
        autonomous_detector.detect_video(input_video)
    else:
        raise NotImplemented


if __name__ == '__main__':
    fire.Fire(main)

