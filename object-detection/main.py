import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.data.catalog import DatasetCatalog
from detectron2.utils.logger import setup_logger

import numpy as np
import cv2
import random
import csv
import os

setup_logger()


def get_object_dicts():
    input_data = {}
    with open('datasets/TJHSST/train_solution_bounding_boxes.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        for idx, row in enumerate(csv_reader):
            if idx == 0:
                continue

            filename = row[0]
            bbox =  row[1:]
            if filename not in input_data:
                input_data[filename] = [bbox]
            else:
                input_data[filename].append(bbox)

    dataset_dicts = []
    for idx, (filename, bboxes) in enumerate(input_data.items()):
        record = {}

        filepath = os.path.join('datasets/TJHSST/train', filename)
        height, width = cv2.imread(filepath).shape[:2]

        record["image_id"] = idx
        record["file_name"] = filepath
        record["height"] = height
        record["width"] = width

        annotations = []
        for bbox in bboxes:
            annotations.append({
                'bbox': [float(b) for b in bbox],
                'bbox_mode': BoxMode.XYXY_ABS,
                'category_id': 0
            })
        record['annotations'] = annotations
        dataset_dicts.append(record)

    return dataset_dicts


if __name__ == '__main__':
    for d in ['train']:
        DatasetCatalog.register(f"car_{d}", lambda d=d: get_object_dicts())
        MetadataCatalog.get(f"car_{d}").set(thing_classes=["car"])
    car_metadata = MetadataCatalog.get("car_train")

    dataset_dicts = get_object_dicts()
    for d in random.sample(dataset_dicts, 3):
        print('d: ', d)
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=car_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow('Car', out.get_image()[:, :, ::-1])
        cv2.waitKey()

