import torch

from PIL import Image
import cv2
import fire
import numpy as np

from object_detection.yolov5.config import global_config as gc
from object_detection.yolov5.utils.general import non_max_suppression
from object_detection.yolov5.utils.torch_utils import time_synchronized


class YoloDetector:
    def __init__(self):
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.classes = None     # filter by class
        self.agnostic_nms = False
        self.max_det = 1000     # maximum detections per image

        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

        # Half precision if using Cuda
        if torch.cuda.is_available():
            self.half_precision = True
            self.device = torch.device('cuda:0')
            self.model.half()
        else:
            self.half_precision = False
            self.device = torch.device('cpu')

        # Speed up constant image size inference
        torch.backends.cudnn.benchmark = True

    def detect(self, img):
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half_precision else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        t2 = time_synchronized()

        



def main():
    yolo_detector = YoloDetector()
    img = cv2.imread(gc.test_image)
    yolo_detector.detect(img)


if __name__ == '__main__':
    fire.Fire(main)

