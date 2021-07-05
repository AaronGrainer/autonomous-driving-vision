import fire
import cv2
import yaml
from collections import OrderedDict
from PIL import Image

import torch
from torchvision.transforms import ToTensor, Compose, Resize
from torch.cuda.amp import autocast

# from road_detection.auto_drive.model import erfnet_resnet
from lane_detection.auto_drive.model import erfnet_resnet

from detectron2.utils.visualizer import Visualizer


class RoadDetector:
    def __init__(self):
        self.dataset = 'cityscapes'
        self.city_aug = 2

        with open('road_detection/auto_drive/configs.yaml', 'r') as f:
            configs = yaml.load(f, Loader=yaml.Loader)

        self.mean = configs['general']['mean']
        self.std = configs['general']['std']

        num_classes = configs[self.dataset]['num_classes']
        pretrained_weights = configs[self.dataset]['pretrained_weights']
        erfnet_model_file = configs[self.dataset]['erfnet_model_file']

        self.mixed_precision = configs[self.dataset]['mixed_precision']
        self.sizes = configs[self.dataset]['sizes']
        self.categories = configs[self.dataset]['categories']
        self.colors = configs[self.dataset]['colors']

        weights = torch.tensor(configs[self.dataset]['weights'])
        self.input_sizes = configs[self.dataset]['input_sizes']

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = erfnet_resnet(pretrained_weights, num_classes=num_classes, num_lanes=0,
                                   dropout_1=0.03, dropout_2=0.3, flattened_size=3965)
        self.model.to(self.device)
        weights = weights.to(self.device)

        self._load_checkpoint(erfnet_model_file)

    def _load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        # To keep BC while having a acceptable variable name for lane detection
        checkpoint['model'] = OrderedDict((k.replace('aux_head', 'lane_classifier') if 'aux_head' in k else k, v)
                                           for k, v in checkpoint['model'].items())
        self.model.load_state_dict(checkpoint['model'])

    @torch.no_grad()
    def _predict(self, img):
        ori_h, ori_w, _ = img.shape
        self.model.eval()

        img_input = Image.fromarray(img)
        transforms = Compose([
            ToTensor(),
            Resize(size=self.input_sizes[0])
        ])
        img_input = transforms(img_input).unsqueeze(0)

        img_input = img_input.to(self.device)
        with autocast(self.mixed_precision):
            output = self.model(img_input)['out']
            output = torch.nn.functional.interpolate(output, size=(ori_h, ori_w), mode='bilinear',
                                                     align_corners=True)
            output = output.argmax(1)

        output = output.cpu().numpy()
        output = output == 0
        
        return output

    def _visualizer(self, img, seg_mask):
        v = Visualizer(img, scale=1)

        v.overlay_instances(masks=seg_mask, assigned_colors=['palegreen'])

        out = v.get_output()
        return out.get_image()[:, :, ::-1]

    def detect(self, img):
        seg_mask = self._predict(img)
        return self._visualizer(img, seg_mask)

    def detect_img(self, img):
        img = self.detect(img)
        cv2.imshow('preds', img)
        cv2.waitKey()

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

