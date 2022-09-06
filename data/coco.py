from pathlib import Path
from typing import Callable, Tuple

import cv2
import numpy as np
from pycocotools.coco import COCO

from data.dataset_wrappers import Dataset


def remove_useless_info(coco):
    """Remove useless info in COCO dataset.

    COCO object is modified inplace.
    This function is mainly used for saving memory (save about 30% mem).
    """
    if isinstance(coco, COCO):
        dataset = coco.dataset
        dataset.pop("info", None)
        dataset.pop("licenses", None)
        for img in dataset["images"]:
            img.pop("license", None)
            img.pop("coco_url", None)
            img.pop("date_captured", None)
            img.pop("flickr_url", None)
        if "annotations" in coco.dataset:
            for anno in coco.dataset["annotations"]:
                anno.pop("segmentation", None)


class COCODataset(Dataset):
    """COCO Dataset class"""

    def __init__(
        self,
        data_dir: str,
        json_file: str = "instances_train2017.json",
        name: str = "train2017",
        img_size: Tuple = (416, 416),
        preproc: Callable = None,
    ):
        """COCO dataset initialization.

        Annnotation data are read into memory by COCO API

        Args:
            data_dir (str, optional): Dataset root directory. Defaults to None.
            json_file (str, optional): COCO json filename. Defaults to "instances_train2017.json".
            name (str, optional): COCO data name (e.g. "train2017", "val2017"). Defaults to "train2017".
            img_size (Tuple, optional): Target image size after pre-processing. Defaults to (416, 416).
            preproc (Callable, optional): Data augmentation strategy. Defaults to None.
        """
        super().__init__(img_size)
        self.data_dir = data_dir
        self.json_file = json_file

        self.coco = COCO(Path(self.data_dir, "annotations", self.json_file))
        remove_useless_info(self.coco)

        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in self.cats])
        self.imgs = None
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self.annotations = self._load_coco_annotations()

    def __len__(self):
        return len(self.ids)

    def __del__(self):
        del self.imgs

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min(width, x1 + np.max((0, obj["bbox"][2])))
            y2 = np.min(height, y1 + np.max((0, obj["bbox"][2])))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 5))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)

        resized_info = (int(height * r), int(width * r))

        file_name = im_ann["file_name"] if "file_name" in im_ann else f"{id_:012}.jpg"

        return (res, img_info, resized_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_image(self, index):
        file_name = self.annotations[index][3]

        img_file = Path(self.data_dir, self.name, file_name)

        img = cv2.imread(img_file)
        assert img is not None, f"Filename {img_file} not found"

        return img

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resize_img = cv2.resize(
            img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR
        ).astype(np.uint8)
        return resize_img

    def pull_item(self, index):
        id_ = self.ids[index]

        res, img_info, resized_info, _ = self.annotations[index]
        if self.imgs is not None:
            pad_img = self.imgs[index]
            img = pad_img[: resized_info[0], : resized_info[1], :].copy()
        else:
            img = self.load_resized_img(index)

        return img, res.copy(), img_info, np.array([id_])
