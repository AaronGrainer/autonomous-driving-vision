import fire
import cv2
import yaml

import torch


class RoadDetector:
    def __init__(self):
        with open('lane_detection/auto_drive/configs.yaml', 'r') as f:
            configs = yaml.load(f, Loader=yaml.Loader)

        self.mean = configs['general']['mean']
        self.std = configs['general']['std']

        num_classes = configs[self.dataset]['num_classes']

        self.sizes = configs[self.dataset]['sizes']
        self.categories = configs[self.dataset]['categories']
        self.colors = configs[self.dataset]['colors']

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def detect_img(self, img):
        pass

    def detect_video(self, input_video):
        pass



def main(detect_type='image'):
    if detect_type == 'image':
        img_path = 'test_asset/usa_laguna_moment.jpg'
        img = cv2.imread(img_path)[:, :, ::-1]

        road_detector = RoadDetector()
        road_detector.detect_img(img)
    elif detect_type == 'video':
        input_video = 'test_asset/usa_laguna.mp4'

        road_detector = RoadDetector()
        road_detector.detect_video(input_video)
    else:
        raise ValueError


if __name__ == '__main__':
    fire.Fire(main)

