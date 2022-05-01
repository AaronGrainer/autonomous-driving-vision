import os

import cv2
import numpy as np
from PIL import Image


def colorEncode(labelmap, colors, mode="RGB"):
    labelmap = labelmap.astype("int")
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3), dtype=np.uint8)
    for label in np.unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * np.tile(
            colors[label], (labelmap.shape[0], labelmap.shape[1], 1)
        )

    if mode == "BGR":
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb


class Vedio:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 15, (1280, 480))

    def addImage(self, img, colorMask):
        img = img[:, :, ::-1]
        colorMask = colorMask[:, :, ::-1]
        img = np.concatenate([img, colorMask], axis=1)
        self.cap.write(img)

    def releaseCap(self):
        self.cap.release()


class Map16:
    def __init__(self, vedioCap, visualpoint=True):
        self.names = (
            "background",
            "floor",
            "bed",
            "cabinet",
            "wardrobe",
            "bookcase",
            "shelf",
            "person",
            "door",
            "table",
            "desk",
            "coffee",
            "chair",
            "armchair",
            "sofa",
            "bench",
            "swivel",
            "stool",
            "rug",
            "railing",
            "column",
            "refrigerator",
            "stairs",
            "stairway",
            "step",
            "escalator",
            "wall",
            "dog",
            "plant",
        )
        self.colors = np.array(
            [
                [0, 0, 0],
                [0, 0, 255],
                [0, 255, 0],
                [0, 255, 255],
                [255, 0, 0],
                [255, 0, 255],
                [255, 255, 0],
                [255, 255, 255],
                [0, 0, 128],
                [0, 128, 0],
                [128, 0, 0],
                [0, 128, 128],
                [128, 0, 0],
                [128, 0, 128],
                [128, 128, 0],
                [128, 128, 128],
                [192, 192, 192],
            ],
            dtype=np.uint8,
        )
        self.outDir = "lane_detection/ddr_net/output/map16"
        self.vedioCap = vedioCap
        self.visualpoint = visualpoint

    def visualize_result(self, data, pred, dir, img_name=None):
        img = data

        pred = np.int32(pred)
        pixs = pred.size
        uniques, counts = np.unique(pred, return_counts=True)
        for idx in np.argsort(counts)[::-1]:
            name = self.names[uniques[idx]]
            ratio = counts[idx] / pixs * 100
            if ratio > 0.1:
                print(f"  {name}: {ratio:.2f}%")

        # calculate point
        if self.visualpoint:
            img = img.copy()
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_gray = np.float32(img_gray)

            goodfeatures_corners = cv2.goodFeaturesToTrack(img_gray, 400, 0.01, 10)
            goodfeatures_corners = np.int0(goodfeatures_corners)

            for i in goodfeatures_corners:
                x, y = i.flatten()
                cv2.circle(
                    img,
                    (x, y),
                    3,
                    [
                        0,
                        255,
                    ],
                    -1,
                )

        # colorize prediction
        pred_color = colorEncode(pred, self.colors).astype(np.uint8)

        im_vis = img * 0.7 + pred_color * 0.3
        im_vis = im_vis.astype(np.uint8)

        # for vedio result show
        self.vedioCap.addImage(im_vis, pred_color)

        img_name = img_name
        if not os.path.exists(dir):
            os.makedirs(dir)
        Image.fromarray(im_vis).save(os.path.join(dir, img_name))
