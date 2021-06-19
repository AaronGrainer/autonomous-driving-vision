import torch
from torchvision import transforms
import os
import cv2
from PIL import Image
import numpy as np
import scipy.special

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


if __name__ == '__main__':
    # lane_detection = LaneDetection()
    # lane_detection.detect_video()

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
        

