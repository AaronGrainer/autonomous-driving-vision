import fire
import cv2
import yaml
from collections import OrderedDict
import warnings
from PIL import Image

import torch
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

        transforms = Compose([
            Resize(size=self.input_sizes[0]),
            ToTensor(),
            Normalize(mean=self.mean, std=self.std)
        ])
        img = transforms(img).unsqueeze(0)

        img = img.to(self.device)
        with autocast(self.mixed_precision):
            output = self.model(img)

            prob_map = torch.nn.functional.interpolate(output['out'], size=self.input_sizes[0], mode='bilinear',
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
        for j in range(existence.shape[0]):
            lane_coordinates = prob_to_lines(prob_map[j], existence[j], resize_shape=self.input_sizes[1],
                                             gap=self.gap, ppl=self.ppl, thresh=self.threshold)
            print('lane_coordinates: ', lane_coordinates)

    def detect_img(self, img):
        self._predict(img)


def main(detect_type='image'):
    if detect_type == 'image':
        img_path = 'test_asset/usa_laguna_moment.jpg'
        # img = cv2.imread(img_path)[:, :, ::-1]
        img = Image.open(img_path).convert('RGB')

        lane_detector = LaneDetector()
        lane_detector.detect_img(img)
    else:
        raise NotImplemented


if __name__ == '__main__':
    fire.Fire(main)

