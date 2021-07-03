import fire
import cv2
import yaml
from collections import OrderedDict
import warnings
from PIL import Image
import matplotlib.pyplot as plt
import time

import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Compose, Resize, Normalize
from torch.cuda.amp import autocast

from lane_detection.auto_drive.model import erfnet_resnet
from lane_detection.auto_drive.utils import prob_to_lines


class LaneDetector:
    def __init__(self):
        self.dataset = 'culane'
        self.backbone = 'erfnet'

        with open('lane_detection/auto_drive/configs.yaml', 'r') as f:
            configs = yaml.load(f, Loader=yaml.Loader)

        self.mean = configs['general']['mean']
        self.std = configs['general']['std']

        num_classes = configs[self.dataset]['num_classes']
        weights = configs[self.dataset]['weights']
        pretrained_weights = configs[self.dataset]['pretrained_weights']
        erfnet_model_file = configs[self.dataset]['erfnet_model_file']

        self.mixed_precision = configs[self.dataset]['mixed_precision']
        self.input_sizes = configs[self.dataset]['input_sizes']
        self.max_lane = configs[self.dataset]['max_lane']
        self.gap = configs[self.dataset]['gap']
        self.ppl = configs[self.dataset]['ppl']
        self.threshold = configs[self.dataset]['threshold']

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = erfnet_resnet(pretrained_weights, num_classes=num_classes)
        self.model.to(self.device)
        weights = torch.tensor(weights).to(self.device)

        self._load_checkpoint(erfnet_model_file)

    def _load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        # To keep BC while having a acceptable variable name for lane detection
        checkpoint['model'] = OrderedDict((k.replace('aux_head', 'lane_classifier') if 'aux_head' in k else k, v)
                                           for k, v in checkpoint['model'].items())
        self.model.load_state_dict(checkpoint['model'])

    @torch.no_grad()
    def _predict(self, img):
        all_lanes = []
        self.model.eval()

        img_input = Image.fromarray(img)
        transforms = Compose([
            Resize(size=self.input_sizes),
            ToTensor(),
            Normalize(mean=self.mean, std=self.std)
        ])
        img_input = transforms(img_input).unsqueeze(0)

        img_input = img_input.to(self.device)
        with autocast(self.mixed_precision):
            output = self.model(img_input)

            prob_map = F.interpolate(output['out'], size=self.input_sizes, mode='bilinear',
                                     align_corners=True).softmax(dim=1)
            existence_conf = output['lane'].sigmoid()
            existence = existence_conf > 0.5

            if self.max_lane != 0:  # Lane max number prior for testing
                # Maybe too slow (but should be faster than topk/sort),
                # consider batch size >> max number of lanes
                while (existence.sum(dim=1) > self.max_lane).sum() > 0:
                    indices = (existence.sum(dim=1, keepdim=True) > self.max_lane).expand_as(existence) * \
                              (existence_conf == existence_conf.min(dim=1, keepdim=True).values)
                    existence[indices] = 0
                    existence_conf[indices] = 1.1  # So we can keep using min

        prob_map = prob_map.cpu().numpy()
        existence = existence.cpu().numpy()

        # Get coordinates for lanes
        ori_h, ori_w, _ = img.shape
        for j in range(existence.shape[0]):
            lane_coordinates = prob_to_lines(prob_map[j], existence[j], resize_shape=(ori_h, ori_w),
                                             gap=self.gap, ppl=self.ppl, thresh=self.threshold)
            all_lanes.extend(lane_coordinates)

        return all_lanes

    def _visualizer(self, img, all_lanes, display_type='dot'):
        img = img.copy()
        for lanes in all_lanes:
            if lanes:
                if display_type == 'dot':
                    for (x, y) in lanes:
                        cv2.circle(img, (int(x), int(y)), radius=3, color=(0, 0, 255), thickness=-1)
                elif display_type == 'line':
                    lanes = [lanes[0], lanes[-1]]
                    pt1, pt2 = [(int(lanes[i][0]), int(lanes[i][1])) for i in range(len(lanes))]
                    cv2.line(img, pt1, pt2, color=(121, 172, 252), thickness=2)
                else:
                    raise ValueError

        return img[:, :, ::-1]

    def detect(self, img):
        all_lanes = self._predict(img)
        return self._visualizer(img, all_lanes, display_type='line')

    def detect_img(self, img):
        img = self.detect(img)
        cv2.imshow('preds', img)
        cv2.waitKey()

    def detect_video(self, input_video):
        cap = cv2.VideoCapture(input_video)
        ret, frame = cap.read()

        ax1 = plt.subplot(111)
        im1 = ax1.imshow(frame)

        plt.ion()
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is True:
                start_time = time.time()
                frame = self.detect(frame)
                im1.set_data(frame)
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
        # img = Image.open(img_path).convert('RGB')

        lane_detector = LaneDetector()
        lane_detector.detect_img(img)
    elif detect_type == 'video':
        input_video = 'test_asset/usa_laguna.mp4'

        lane_detector = LaneDetector()
        lane_detector.detect_video(input_video)
    else:
        raise ValueError


if __name__ == '__main__':
    fire.Fire(main)

