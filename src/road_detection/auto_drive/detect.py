import time
from collections import OrderedDict

import cv2
import matplotlib.pyplot as plt
import torch
import typer
import yaml
from detectron2.utils.visualizer import Visualizer
from PIL import Image
from torch.cuda.amp import autocast
from torchvision.transforms import Compose, Resize, ToTensor

# from src.road_detection.auto_drive.model import erfnet_resnet
from src.lane_detection.auto_drive.model import erfnet_resnet

app = typer.Typer()


class RoadDetector:
    def __init__(self):
        self.dataset = "cityscapes"
        self.city_aug = 2

        with open("road_detection/auto_drive/configs.yaml") as f:
            configs = yaml.load(f, Loader=yaml.Loader)

        self.mean = configs["general"]["mean"]
        self.std = configs["general"]["std"]

        num_classes = configs[self.dataset]["num_classes"]
        pretrained_weights = configs[self.dataset]["pretrained_weights"]
        erfnet_model_file = configs[self.dataset]["erfnet_model_file"]

        self.mixed_precision = configs[self.dataset]["mixed_precision"]
        self.sizes = configs[self.dataset]["sizes"]
        self.categories = configs[self.dataset]["categories"]
        self.colors = configs[self.dataset]["colors"]

        weights = torch.tensor(configs[self.dataset]["weights"])
        self.input_sizes = configs[self.dataset]["input_sizes"]

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = erfnet_resnet(
            pretrained_weights,
            num_classes=num_classes,
            num_lanes=0,
            dropout_1=0.03,
            dropout_2=0.3,
            flattened_size=3965,
        )
        self.model.to(self.device)
        weights = weights.to(self.device)

        self._load_checkpoint(erfnet_model_file)

    def _load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        # To keep BC while having a acceptable variable name for lane detection
        checkpoint["model"] = OrderedDict(
            (k.replace("aux_head", "lane_classifier") if "aux_head" in k else k, v)
            for k, v in checkpoint["model"].items()
        )
        self.model.load_state_dict(checkpoint["model"])

    @torch.no_grad()
    def _predict(self, img):
        ori_h, ori_w, _ = img.shape
        self.model.eval()

        img_input = Image.fromarray(img)
        transforms = Compose([ToTensor(), Resize(size=self.input_sizes[0])])
        img_input = transforms(img_input).unsqueeze(0)

        img_input = img_input.to(self.device)
        with autocast(self.mixed_precision):
            output = self.model(img_input)["out"]
            output = torch.nn.functional.interpolate(
                output, size=(ori_h, ori_w), mode="bilinear", align_corners=True
            )
            output = output.argmax(1)

        output = output.cpu().numpy()
        output = output == 0

        return output

    def _visualizer(self, img, road_seg):
        v = Visualizer(img, scale=1)

        v.overlay_instances(masks=road_seg, assigned_colors=["palegreen"])

        out = v.get_output()
        return out.get_image()[:, :, ::-1]

    def detect(self, img):
        road_seg = self._predict(img)
        return self._visualizer(img, road_seg)

    def detect_img_road(self, img):
        road_seg = self._predict(img)
        return road_seg

    def detect_img(self, img):
        img = self.detect(img)
        cv2.imshow("preds", img)
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


@app.command()
def main(detect_type="image"):
    if detect_type == "image":
        img_path = "test_asset/usa_laguna_moment.jpg"
        img = cv2.imread(img_path)[:, :, ::-1]

        road_detector = RoadDetector()
        road_detector.detect_img(img)
    elif detect_type == "video":
        input_video = "test_asset/usa_laguna.mp4"

        road_detector = RoadDetector()
        road_detector.detect_video(input_video)
    else:
        raise ValueError


if __name__ == "__main__":
    app()
