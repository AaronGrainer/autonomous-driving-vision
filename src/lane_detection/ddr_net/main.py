import os
import random
import timeit

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms

from src.lane_detection.ddr_net.ddrnet_23_slim import get_seg_model
from src.lane_detection.ddr_net.utils import Map16, Vedio


def pad_image(image, h, w, size, padvalue):
    pad_image = image.copy()
    pad_h = max(size[0] - h, 0)
    pad_w = max(size[1] - w, 0)
    if pad_h > 0 or pad_w > 0:
        pad_image = cv2.copyMakeBorder(
            image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=padvalue
        )

    return pad_image


def rand_crop(image, label):
    crop_size = (512, 1024)
    ignore_label = -1

    h, w = image.shape[:-1]
    image = pad_image(image, h, w, crop_size, (0.0, 0.0, 0.0))
    label = pad_image(label, h, w, crop_size, (ignore_label,))

    new_h, new_w = label.shape
    x = random.randint(0, new_w - crop_size[1])
    y = random.randint(0, new_h - crop_size[0])
    image = image[y : y + crop_size[0], x : x + crop_size[1]]
    label = label[y : y + crop_size[0], x : x + crop_size[1]]

    return image, label


def multi_scale_aug(image, label=None, rand_scale=1, rand_crop=True):
    print("image: ", image, image.shape)
    base_size = 2048

    long_size = np.int64(base_size * rand_scale + 0.5)
    h, w = image.shape[:2]
    print("h, w: ", h, w)
    if h > w:
        new_w = np.int64(w * long_size / h + 0.5)
        new_h = long_size
    else:
        new_w = long_size
        new_h = np.int64(h * long_size / w + 0.5)
    print("new_w: ", new_w)
    print("new_h: ", new_h)

    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if label is not None:
        label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    else:
        return image

    if rand_crop:
        image, label = rand_crop(image, label)

    return image, label


def inference(model, image, flip=False):
    NUM_OUTPUTS = 2
    OUTPUT_INDEX = 1
    ALIGN_CORNERS = False

    size = image.size()
    pred = model(image)

    if NUM_OUTPUTS > 1:
        pred = pred[OUTPUT_INDEX]

    pred = F.interpolate(input=pred, size=size[-2:], mode="bilinear", align_corners=ALIGN_CORNERS)

    if flip:
        flip_img = image.numpy()[:, :, :, ::-1]
        flip_output = model(torch.from_numpy(flip_img.copy()))

        if NUM_OUTPUTS > 1:
            flip_output = flip_output[OUTPUT_INDEX]

        flip_output = F.interpolate(
            input=flip_output, size=size[-2:], mode="bilinear", align_corners=ALIGN_CORNERS
        )

        flip_pred = flip_output.cpu().numpy().copy()
        flip_pred = torch.from_numpy(flip_pred[:, :, :, ::-1].copy()).cuda()
        pred += flip_pred
        pred = pred * 0.5
    return pred.exp()


def multi_scale_inference(model, image, scales=[1], flip=False):
    ALIGN_CORNERS = False

    batch, _, ori_height, ori_width = image.size()
    assert batch == 1, "only supporting batchsize 1."
    # image = image.numpy()[0].transpose((1, 2, 0)).copy()
    image = image.squeeze(0).numpy().transpose((1, 2, 0)).copy()

    num_classes = 19
    crop_size = (512, 1024)

    stride_h = np.int(crop_size[0] * 1.0)
    stride_w = np.int(crop_size[1] * 1.0)

    final_pred = torch.zeros([1, num_classes, ori_height, ori_width]).cuda()

    for scale in scales:
        new_img = multi_scale_aug(image=image, rand_scale=scale, rand_crop=False)
        height, width = new_img.shape[:-1]

        if scale <= 1.0:
            new_img = new_img.transpose((2, 0, 1))
            new_img = np.expand_dims(new_img, axis=0)
            new_img = torch.from_numpy(new_img).cuda()
            preds = inference(model, new_img, flip)
            preds = preds[:, :, 0:height, 0:width]

        preds = F.interpolate(
            preds, (ori_height, ori_width), mode="bilinear", align_corners=ALIGN_CORNERS
        )
        final_pred += preds
    return final_pred


def test(model, sv_dir="", sv_pred=True):
    SCALE_LIST = [1]
    FLIP_TEST = False
    ALIGN_CORNERS = False

    vedioCap = Vedio("lane_detection/ddr_net/output/cdOffice.mp4")
    map16 = Map16(vedioCap)

    with torch.no_grad():
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        image = cv2.imread("test_asset/solidWhiteCurve.jpg", cv2.IMREAD_COLOR)
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= mean
        image /= std
        # image = image.transpose((2, 0, 1))
        size = np.array(image.shape)

        image = transforms.ToTensor()(image).unsqueeze(0)
        pred = multi_scale_inference(model, image, scales=SCALE_LIST, flip=FLIP_TEST)

        if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
            pred = F.interpolate(pred, size[-2:], mode="bilinear", align_corners=ALIGN_CORNERS)

        if sv_pred:
            # mean=[0.485, 0.456, 0.406],
            # std=[0.229, 0.224, 0.225]
            image = image.squeeze(0)
            image = image.numpy().transpose((1, 2, 0))
            image *= [0.229, 0.224, 0.225]
            image += [0.485, 0.456, 0.406]
            image *= 255.0
            image = image.astype(np.uint8)

            _, pred = torch.max(pred, dim=1)
            pred = pred.squeeze(0).cpu().numpy()
            map16.visualize_result(image, pred, sv_dir, "solidWhiteCurve.jpg")
            # sv_path = os.path.join(sv_dir, 'test_results')
            # if not os.path.exists(sv_path):
            #     os.mkdir(sv_path)
            # test_dataset.save_pred(image, pred, sv_path, name)

    vedioCap.releaseCap()


def main():
    # Build model
    # module = eval('models.ddrnet_23_slim')
    # module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = get_seg_model()
    model.cuda()
    model.eval()

    dump_input = torch.rand((1, 3, 1024, 1024))

    dump_output = model(dump_input.cuda())
    # print('dump_output: ', dump_output)

    test(model, sv_dir="lane_detection/ddr_net/output/map16")


if __name__ == "__main__":
    main()
