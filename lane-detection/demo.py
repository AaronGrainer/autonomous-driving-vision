import torch
from torchvision import transforms
import os
import cv2
from PIL import Image
import numpy as np
import scipy.special

from model.model import ParsingNet
from data.dataset import LaneTestDataset
from data.constant import culane_row_anchor
from config import global_config as gc


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    assert gc.backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']
    
    cls_num_per_lane = 18

    net = ParsingNet(pretrained=False,
                     backbone=gc.backbone, 
                     cls_dim=(gc.griding_num+1, cls_num_per_lane, gc.num_lanes),
                     use_aux=False).cuda()
    # Don't need auxiliary segmentation during testing

    state_dict = torch.load(gc.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    if not os.path.exists(gc.output_dir):
        os.mkdir(gc.output_dir)
    
    img_transform = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    img_w, img_h = 1640, 590

    # img = Image.open(gc.test_img)
    # img = img_transforms(img)

    # imgs = img.unsqueeze(0).cuda()
    # with torch.no_grad():
    #     out = net(imgs)
    # out = torch.squeeze(out, 0)
    # out = out.cpu().numpy()
    # cv2.imshow('Prediction', out)
    # cv2.waitKey()

    dataset = LaneTestDataset(gc.data_root, gc.list_path, img_transform=img_transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_name = os.path.join(gc.output_dir, 'test0_normal.avi')
    vout = cv2.VideoWriter(video_name, fourcc, 30.0, (img_w, img_h))

    for i, data in enumerate(loader):
        imgs, names = data
        imgs = imgs.cuda()
        with torch.no_grad():
            out = net(imgs)
        
        col_sample = np.linspace(0, 800 - 1, gc.griding_num)
        col_sample_w = col_sample[1] - col_sample[0]

        out_j = out[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :]
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        idx = np.arange(gc.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob*idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == gc.griding_num] = 0
        out_j = loc
        
        vis = cv2.imread(os.path.join(gc.data_root, names[0]))
        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (culane_row_anchor[cls_num_per_lane - 1 - k] / 288)) - 1)
                        cv2.circle(vis, ppp, 5, (0, 255, 0), -1)
        vout.write(vis)

    vout.release()
        

