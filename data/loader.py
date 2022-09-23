import random
import uuid

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader as torchDataLoader

from common import config
from data.coco import COCODataset
from data.data_augment import TrainTransform
from data.mosaic_detection import MosaicDetection
from data.samplers import InfiniteSampler, YoloBatchSampler


class COCODataModule(LightningDataModule):
    def __init__(self):
        pass

    def setup(self, no_aug=False):
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

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=config.YOLOX_CONFIG["flip_prob"],
                hsv_prob=config.YOLOX_CONFIG["hsv_prob"],
            ),
            degrees=config.YOLOX_CONFIG["degrees"],
            translate=config.YOLOX_CONFIG["translate"],
            mosaic_scale=config.YOLOX_CONFIG["mosaic_scale"],
            mixup_scale=config.YOLOX_CONFIG["mixup_scale"],
            shear=config.YOLOX_CONFIG["shear"],
            enable_mixup=config.YOLOX_CONFIG["enable_mixup"],
            mosaic_prob=config.YOLOX_CONFIG["mosaic_prob"],
            mixup_prob=config.YOLOX_CONFIG["mixup_prob"],
        )

        self.dataset = dataset

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=config.YOLOX_CONFIG["batch_size"],
            drop_last=False,
            mosaic=not no_aug,
        )

        self.dataloader_kwargs = {
            "num_workers": config.YOLOX_CONFIG["data_num_workers"],
            "pin_memory": True,
            "batch_sampler": batch_sampler,
            "worker_init_fn": worker_init_reset_seed,
        }

    def train_dataloader(self):
        return DataLoader(self.dataset, **self.dataloader_kwargs)


class DataLoader(torchDataLoader):
    """Lightnet dataloader that enables on the fly resizing of the images.

    See :class:`torch.utils.data.DataLoader` for more information on the arguments.
    Check more on the following website:
    https://gitlab.com/EAVISE/lightnet/-/blob/master/lightnet/data/_dataloading.py
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__initialized = False
        shuffle = False
        batch_sampler = None
        if len(args) > 5:
            shuffle = args[2]
            sampler = args[3]
            batch_sampler = args[4]
        elif len(args) > 4:
            shuffle = args[2]
            sampler = args[3]
            if "batch_sampler" in kwargs:
                batch_sampler = kwargs["batch_sampler"]
        elif len(args) > 3:
            shuffle = args[2]
            if "sampler" in kwargs:
                sampler = kwargs["sampler"]
            if "batch_sampler" in kwargs:
                batch_sampler = kwargs["batch_sampler"]
        else:
            if "shuffle" in kwargs:
                shuffle = kwargs["shuffle"]
            if "sampler" in kwargs:
                sampler = kwargs["sampler"]
            if "batch_sampler" in kwargs:
                batch_sampler = kwargs["batch_sampler"]

        # Use custom BatchSampler
        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = torch.utils.data.sampler.RandomSampler(self.dataset)
                else:
                    sampler = torch.utils.data.sampler.SequentialSampler(self.dataset)
            batch_sampler = YoloBatchSampler(
                sampler, self.batch_size, self.drop_last, input_dimension=self.dataset.input_dim
            )

        self.batch_sampler = batch_sampler

        self.__initialized = True

    def close_mosaic(self):
        self.batch_sampler.mosaic = False


def worker_init_reset_seed(worker_id):
    seed = uuid.uuid4().int % 2**32
    random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    np.random.seed(seed)
