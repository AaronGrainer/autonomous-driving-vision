import os
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import fire
import cv2
import matplotlib.pyplot as plt
import time

import torch
from torchvision import transforms

import depth_estimation.monodepth.networks as networks
from depth_estimation.monodepth.layers import disp_to_depth
from depth_estimation.monodepth.utils import download_model_if_doesnt_exist


STEREO_SCALE_FACTOR = 5.4


class MonoDepthEstimator:
    def __init__(self, dir_path='depth_estimation/monodepth/models', model_name='mono+stereo_640x192'):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        download_model_if_doesnt_exist(dir_path, model_name)
        model_path = os.path.join(dir_path, model_name)
        encoder_path = os.path.join(model_path, "encoder.pth")
        depth_decoder_path = os.path.join(model_path, "depth.pth")

        # Load pretrained encoder
        self.encoder = networks.ResnetEncoder(18, False)
        loaded_dict_enc = torch.load(encoder_path, map_location=self.device)

        # Extract the height and width of image that this model was trained with
        self.feed_height = loaded_dict_enc['height']
        self.feed_width = loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)
        self.encoder.to(self.device)
        self.encoder.eval()

        # Load pretrained decoder
        self.depth_decoder = networks.DepthDecoder(
            num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

        loaded_dict = torch.load(depth_decoder_path, map_location=self.device)
        self.depth_decoder.load_state_dict(loaded_dict)

        self.depth_decoder.to(self.device)
        self.depth_decoder.eval()

    def _predict(self, img):
        with torch.no_grad():
            # Load image and preprocess
            input_image = cv2.resize(img, (self.feed_width, self.feed_height))
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # Prediction
            input_image = input_image.to(self.device)
            features = self.encoder(input_image)
            outputs = self.depth_decoder(features)
        
        return outputs

    def _get_depth_metrics(self, disp):
        scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
        metric_depth = STEREO_SCALE_FACTOR * depth.cpu().numpy()
        
        return metric_depth

    def _get_bbox_depth(self, metric_depth, preds, class_only):
        object_depths = []
        for _, pred in preds.iterrows():
            if pred['name'] in class_only:
                box_depth = metric_depth[0][0][int(pred['ymin']):int(pred['ymax']), int(pred['xmin']):int(pred['xmax'])]
                object_depth = box_depth.mean()
                object_depths.append(object_depth)
            else:
                object_depths.append(0)
        preds['depth'] = object_depths
        
        return preds

    def _visualizer(self, disp):
        """Display colormapped depth image
        """
        disp_resized_np = disp.squeeze().cpu().numpy()
        vmax = np.percentile(disp_resized_np, 95)
        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)

        return colormapped_im

    def detect(self, img, get_depth=False):
        original_height, original_width, _ = img.shape

        outputs = self._predict(img)

        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
            disp, (original_height, original_width), mode="bilinear", align_corners=False)

        if get_depth:
            return self._get_depth_metrics(disp_resized)
        else:
            return self._visualizer(disp_resized)

    def detect_img(self, img):
        colormapped_im = self.detect(img)
        cv2.imshow('preds', colormapped_im)
        cv2.waitKey()

    def detect_img_object(self, img, preds, class_only=None):
        depth_metrics = self.detect(img, get_depth=True)
        preds = self._get_bbox_depth(depth_metrics, preds, class_only)
        return preds

    def detect_video(self, input_video):
        cap = cv2.VideoCapture(input_video)
        ret, frame = cap.read()
        frame_pred = self.detect(frame[:, :, ::-1])

        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)

        im1 = ax1.imshow(frame[:, :, ::-1])
        im2 = ax2.imshow(frame_pred[:, :, ::-1])

        plt.ion()
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is True:
                start_time = time.time()
                frame_pred = self.detect(frame[:, :, ::-1])
                im1.set_data(frame[:, :, ::-1])
                im2.set_data(frame_pred[:, :, ::-1])
                plt.pause(0.001)
                # print("FPS: ", 1.0 / (time.time() - start_time))
            else:
                break
        plt.ioff()
        plt.show()


def main(detect_type='image'):
    if detect_type == 'image':
        img_path = 'test_asset/usa_laguna_moment.jpg'
        img = cv2.imread(img_path)[:, :, ::-1]

        mono_depth_estimator = MonoDepthEstimator()
        mono_depth_estimator.detect_img(img)
    elif detect_type == 'image_bbox':
        img_path = 'test_asset/usa_laguna_moment.jpg'
        img = cv2.imread(img_path)[:, :, ::-1]

        mono_depth_estimator = MonoDepthEstimator()
        mono_depth_estimator.detect_img_object(img)
    elif detect_type == 'video':
        input_video = 'test_asset/usa_laguna.mp4'

        mono_depth_estimator = MonoDepthEstimator()
        mono_depth_estimator.detect_video(input_video)
    else:
        raise ValueError


if __name__ == '__main__':
    fire.Fire(main)

