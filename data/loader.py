from pytorch_lightning import LightningDataModule

from common import config
from data.coco import COCODataset
from data.data_augment import TrainTransform


class COCODataModule(LightningDataModule):
    def __init__(self):
        pass

    def setup(self):
        dataset = COCODataset(
            data_dir=config.YOLOX_CONFIG["data_dir"],
            json_file=config.YOLOX_CONFIG["train_ann"],
            img_size=config.YOLOX_CONFIG["input_size"],
            prepro=TrainTransform(
                max_labels=50,
                flip_prob=config.YOLOX_CONFIG["flip_prob"],
                hsv_prob=config.YOLOX_CONFIG["hsv_prob"],
            ),
        )
