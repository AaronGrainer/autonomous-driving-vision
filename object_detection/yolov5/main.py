import torch

from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances, Boxes

import cv2
import pandas as pd


def main():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    img_path = 'test_asset/solidWhiteCurve.jpg'

    img = cv2.imread(img_path)[:, :, ::-1]

    results = model(img)
    classes = results.names
    predictions = results.pandas().xyxy[0]

    MetadataCatalog.get(f"yolov5").set(thing_classes=classes)
    yolov5_metadata = MetadataCatalog.get("yolov5")

    v = Visualizer(img,
                   metadata=yolov5_metadata,
                   scale=0.5
    )

    for _, pred in predictions.iterrows():
        box_coord = (pred['xmin'], pred['ymin'], pred['xmax'], pred['ymax'])
        v.draw_box(box_coord, edge_color='moccasin')
    out = v.get_output()
    cv2.imshow('Pred', out.get_image()[:, :, ::-1])
    cv2.waitKey()


if __name__ == '__main__':
    main()

