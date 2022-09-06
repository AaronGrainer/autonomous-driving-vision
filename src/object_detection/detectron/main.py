import csv
import os
import random

import cv2
import numpy as np

import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer

setup_logger()


def get_object_dicts():
    input_data = {}
    with open("object_detection/datasets/TJHSST/train_solution_bounding_boxes.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        for idx, row in enumerate(csv_reader):
            if idx == 0:
                continue

            filename = row[0]
            bbox = row[1:]
            if filename not in input_data:
                input_data[filename] = [bbox]
            else:
                input_data[filename].append(bbox)

    dataset_dicts = []
    for idx, (filename, bboxes) in enumerate(input_data.items()):
        record = {}

        filepath = os.path.join("object_detection/datasets/TJHSST/train", filename)
        height, width = cv2.imread(filepath).shape[:2]

        record["image_id"] = idx
        record["file_name"] = filepath
        record["height"] = height
        record["width"] = width

        annotations = []
        for bbox in bboxes:
            annotations.append(
                {"bbox": [float(b) for b in bbox], "bbox_mode": BoxMode.XYXY_ABS, "category_id": 0}
            )
        record["annotations"] = annotations
        dataset_dicts.append(record)

    return dataset_dicts


if __name__ == "__main__":
    for d in ["train"]:
        DatasetCatalog.register(f"car_{d}", lambda d=d: get_object_dicts())
        MetadataCatalog.get(f"car_{d}").set(thing_classes=["car"])
    car_metadata = MetadataCatalog.get("car_train")

    # dataset_dicts = get_object_dicts()
    # for d in random.sample(dataset_dicts, 10):
    #     print('d: ', d)
    #     img = cv2.imread(d["file_name"])
    #     visualizer = Visualizer(img[:, :, ::-1], metadata=car_metadata, scale=0.5)
    #     out = visualizer.draw_dataset_dict(d)
    #     cv2.imshow('Car', out.get_image()[:, :, ::-1])
    #     cv2.waitKey()

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("car_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    )  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.0125  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        256  # faster, and good enough for this toy dataset (default: 512)
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    cfg.OUTPUT_DIR = os.path.join(os.getcwd(), "object_detection/detectron/output")

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    # trainer.train()

    cfg.MODEL.WEIGHTS = os.path.join(
        cfg.OUTPUT_DIR, "model_final.pth"
    )  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    dataset_dicts = get_object_dicts()
    for d in random.sample(dataset_dicts, 1):
        im = cv2.imread(d["file_name"])
        outputs = predictor(
            im
        )  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(
            im[:, :, ::-1],
            metadata=car_metadata,
            scale=0.5,
            instance_mode=ColorMode.IMAGE,  # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        print('outputs["instances"]: ', outputs["instances"])
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow("Ballon Prediction", out.get_image()[:, :, ::-1])
        cv2.waitKey()
