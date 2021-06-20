import torch
from torchvision import transforms
import os
import cv2
from PIL import Image
import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import HuberRegressor, Ridge
import fire

from lane_detection.model.model import ParsingNet
from lane_detection.data.dataset import LaneTestDataset
from lane_detection.data.constant import culane_row_anchor
from lane_detection.config import global_config as gc


class LaneDetection:
    def __init__(self, load_dataloader=True):
        torch.backends.cudnn.benchmark = True

        self.cls_num_per_lane = 18

        self.img_transform = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.img_w, self.img_h = 1640, 590

        self._initialize()
        if load_dataloader:
            self._initialize_dataloader()
        self._initialize_model()

        col_sample = np.linspace(0, 800 - 1, gc.griding_num)
        self.col_sample_w = col_sample[1] - col_sample[0]

    def _initialize(self):
        if not os.path.exists(gc.output_dir):
            os.mkdir(gc.output_dir)

    def _initialize_dataloader(self):
        dataset = LaneTestDataset(gc.data_root, gc.list_path, img_transform=self.img_transform)
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    def _initialize_model(self):
        self.net = ParsingNet(pretrained=False,
                              backbone=gc.backbone, 
                              cls_dim=(gc.griding_num+1, self.cls_num_per_lane, gc.num_lanes),
                              use_aux=False).cuda()

        state_dict = torch.load(gc.test_model, map_location='cpu')['model']
        compatible_state_dict = {}
        for k, v in state_dict.items():
            if 'module.' in k:
                compatible_state_dict[k[7:]] = v
            else:
                compatible_state_dict[k] = v

        self.net.load_state_dict(compatible_state_dict, strict=False)
        self.net.eval()

    def _format_pred_output(self, out):
        out_j = out[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :]
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        idx = np.arange(gc.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob*idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == gc.griding_num] = 0

        return loc

    def _draw_lane_prediction(self, vis, out_j):
        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        ppp = (int(out_j[k, i] * self.col_sample_w * self.img_w / 800) - 1,
                                int(self.img_h * (culane_row_anchor[self.cls_num_per_lane - 1 - k] / 288)) - 1)
                        cv2.circle(vis, ppp, 5, (0, 255, 0), -1)

    def detect(self, ori_img):
        img = Image.fromarray(ori_img)
        img = self.img_transform(img)
        img = img.unsqueeze(0).cuda()

        with torch.no_grad():
            out = self.net(img)

        out_j = self._format_pred_output(out)
        self._draw_lane_prediction(ori_img, out_j)

    def detect_video(self):
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video_name = os.path.join(gc.output_dir, 'test0_normal.avi')
        vout = cv2.VideoWriter(video_name, fourcc, 30.0, (self.img_w, self.img_h))

        for data in self.loader:
            imgs, names = data
            imgs = imgs.cuda()
            with torch.no_grad():
                out = self.net(imgs)

            out_j = self._format_pred_output(out)
            
            vis = cv2.imread(os.path.join(gc.data_root, names[0]))
            self._draw_lane_prediction(vis, out_j)

            vout.write(vis)

        vout.release()


class LaneDetectionCV:
    def __init__(self):
        self.gaussian_kernel = 5
        self.low_threshold, self.high_threshold = [200, 300]
        self.hough_settings = {
            'rho': 1,
            'theta': math.pi/180,
            'threshold': 15,
            'min_line_len': 30,
            'max_line_gap': 40
        }
        self.line_threshold = 330
        self.line_thickness = 6

    def _grayscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def _gaussian_blur(self, img):
        return cv2.GaussianBlur(img, (self.gaussian_kernel, self.gaussian_kernel), 0)

    def _canny(self, img):
        return cv2.Canny(img, self.low_threshold, self.high_threshold)

    def _region_of_interest(self, img, vertices):
        mask = np.zeros_like(img)
        if len(img.shape) > 2:
            channel_count = img.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        cv2.fillPoly(mask, vertices, ignore_mask_color)
        return cv2.bitwise_and(img, mask)

    def _draw_lines(self, img, lines):
        line_dict = {'left': [], 'right': []}
        img_center = img.shape[1] // 2
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x1 < img_center and x2 < img_center:
                    position = 'left'
                elif x1 > img_center and x2 > img_center:
                    position = 'right'
                else:
                    continue
                line_dict[position].append(np.array([x1, y1]))
                line_dict[position].append(np.array([x2, y2]))

        for position, lines_dir in line_dict.items():
            if lines_dir:
                data = np.array(lines_dir)
                data = data[data[:, 1] >= np.array(self.line_threshold)-1]
                x, y = data[:, 0].reshape((-1, 1)), data[:, 1]

                try:
                    model = HuberRegressor(fit_intercept=True, alpha=0.0, max_iter=100, epsilon=1.9)
                    model.fit(x, y)
                except ValueError:
                    model = Ridge(fit_intercept=True, alpha=0.0, random_state=0, normalize=True)
                    model.fit(x, y)

                epsilon = 1e-10
                y1 = np.array(img.shape[0])
                x1 = (y1 - model.intercept_) / (model.coef_ + epsilon)
                y2 = np.array(self.line_threshold)
                x2 = (y2 - model.intercept_) / (model.coef_ + epsilon)
                x = np.append(x, [x1, x2], axis=0)

                y_pred = model.predict(x)
                data = np.append(x, y_pred.reshape((-1, 1)), axis=1)
                cv2.polylines(img, np.int32([data]), isClosed=False, color=(255, 0, 0),
                              thickness=self.line_thickness)

    def _hough_lines(self, img):
        lines = cv2.HoughLinesP(img, rho=self.hough_settings['rho'],
                                theta=self.hough_settings['theta'],
                                threshold=self.hough_settings['threshold'],
                                lines=np.array([]),
                                minLineLength=self.hough_settings['min_line_len'],
                                maxLineGap=self.hough_settings['max_line_gap'])
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        self._draw_lines(line_img, lines)
        return line_img

    def _weighted_img(self, img, initial_img, α=0.95, β=1., γ=0.):
        return cv2.addWeighted(initial_img, α, img, β, γ)

    def detect(self, img_path):
        """Detect lines and draw them on the image
        """
        img = cv2.imread(img_path)
        img_line = self._grayscale(img)
        img_line = self._gaussian_blur(img_line)
        img_line = self._canny(img_line)
        vertices = np.array([[
            (0, img.shape[0]),
            (img.shape[1], img.shape[0]),
            (400, 260),
            (600, 260)
        ]])
        img_line = self._region_of_interest(img_line, vertices)
        img_line = self._hough_lines(img_line)
        return self._weighted_img(img_line, img)


def main(detect_type='cv'):
    if detect_type == 'dataloader':
        lane_detection = LaneDetection()
        lane_detection.detect_video()
    elif detect_type == 'video':
        lane_detection = LaneDetection(load_dataloader=False)

        cap = cv2.VideoCapture(os.path.join(os.getcwd(), 'video.mp4'))
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is True:
                lane_detection.detect(frame)
                cv2.imshow('Frame', frame)
                cv2.waitKey(0)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
    elif detect_type == 'image':
        lane_detection = LaneDetection(load_dataloader=False)

        img = cv2.imread(gc.test_img)
        lane_detection.detect(img)
        cv2.imshow('Image', img)
        cv2.waitKey(0)
    elif detect_type == 'cv':
        lane_detection_cv = LaneDetectionCV()
        img = lane_detection_cv.detect(gc.test_img)
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    fire.Fire(main)

