import torch

from PIL import Image
import cv2
import fire
import numpy as np

from object_detection.yolov5.config import global_config as gc


class YoloDetector:
    def __init__(self):
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
        print('pred: ', pred)



def main():
    yolo_detector = YoloDetector()
    img = cv2.imread(gc.test_image)
    yolo_detector.detect(img)


if __name__ == '__main__':
    fire.Fire(main)

