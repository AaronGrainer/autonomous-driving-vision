from data.dataset_wrappers import Dataset
from typing import Tuple, Callable
from pathlib import Path
from pycocotools.coco import COCO


class COCODataset(Dataset):
    """COCO Dataset class"""
    def __init__(
        self,
        data_dir: str,
        json_file: str = "instances_train2017.json",
        name: str = "train2017",
        img_size: Tuple = (416, 416),
        preproc: Callable = None,
        cache: bool = False
    ):
        """COCO dataset initialization.

        Annnotation data are read into memory by COCO API

        Args:
            data_dir (str, optional): Dataset root directory. Defaults to None.
            json_file (str, optional): COCO json filename. Defaults to "instances_train2017.json".
            name (str, optional): COCO data name (e.g. "train2017", "val2017"). Defaults to "train2017".
            img_size (Tuple, optional): Target image size after pre-processing. Defaults to (416, 416).
            preproc (Callable, optional): Data augmentation strategy. Defaults to None.
            cache (bool, optional): Cache the results. Defaults to False.
        """
        super().__init__(img_size)
        self.data_dir = data_dir
        self.json_file = json_file

        self.coco = COCO(Path(self.data_dir, "annotations", self.json_file))



