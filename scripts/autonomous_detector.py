import time

import cv2
import matplotlib.pyplot as plt

from detectron2.utils.visualizer import Visualizer
from src.depth_estimation.monodepth.detect import MonoDepthEstimator
from src.lane_detection.auto_drive.detect import LaneDetector
from src.object_detection.utils.trafic_light import detect_trafic_light_color
from src.object_detection.yolov5.detect import YoloDetector
from src.road_detection.auto_drive.detect import RoadDetector


class AutonomousDetector:
    def __init__(self):
        self.yolo_detector = YoloDetector()
        self.mono_depth_estimator = MonoDepthEstimator()
        self.lane_detector = LaneDetector()
        self.road_detector = RoadDetector()

    def _visualize(self, img, preds, all_lanes, road_seg):
        v = Visualizer(img, scale=1)

        for _, pred in preds.iterrows():
            box_coord = (pred["xmin"], pred["ymin"], pred["xmax"], pred["ymax"])
            v.draw_box(box_coord, edge_color="moccasin")

            classname = pred["name"].title()
            if classname == "Car":
                text = f"{classname} ({pred['depth']:.1f}m)\n" f"ID: {pred['class_unique_id']}"
            elif classname == "Traffic Light":
                traffic_color = (
                    f"({pred['traffic_color']})" if pred["traffic_color"] != "other" else ""
                )
                text = f"{classname} {traffic_color}\n" f"ID: {pred['class_unique_id']}"
            else:
                text = f"{classname}\n" f"ID: {pred['class_unique_id']}"

            v.draw_text(
                text,
                (pred["xmin"], max(0, pred["ymin"] - 16)),
                font_size=5,
                color="moccasin",
                horizontal_alignment="left",
            )

        for lanes in all_lanes:
            if lanes:
                # Trivially limit lanes to center lane
                if lanes[0][0] >= 250 and lanes[0][0] <= 1050:
                    lane_pts = list(zip(*lanes))
                    v.draw_line(lane_pts[0], lane_pts[1], color="cornflowerblue")

        v.overlay_instances(masks=road_seg, assigned_colors=["palegreen"])

        out = v.get_output()
        return out.get_image()[:, :, ::-1]

    def detect(self, img):
        preds = self.yolo_detector.detect(img, visualize=False, return_preds=True)
        preds = self.mono_depth_estimator.detect_img_object(img, preds, class_only=["car"])
        preds = detect_trafic_light_color(img, preds)
        all_lanes = self.lane_detector.detect_img_lanes(img)
        road_seg = self.road_detector.detect_img_road(img)
        return self._visualize(img, preds, all_lanes, road_seg)

    def detect_img(self, img):
        img_pred = self.detect(img)
        cv2.imshow("preds", img_pred)
        cv2.waitKey()

    def detect_video(self, input_video, frame_skip):
        cap = cv2.VideoCapture(input_video)
        ret, frame = cap.read()

        if frame_skip:
            for _ in range(frame_skip):
                _, _ = cap.read()

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
