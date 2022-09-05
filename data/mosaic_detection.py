import random
from data.dataset_wrappers import Dataset
from typing import Callable, Tuple


class MosaicDetection(Dataset):
    """Detection dataset wrapper that performs mixup for normal dataset."""
    
    def __init__(
        self,
        dataset,
        img_size,
        mosaic: bool = True,
        preproc: Callable = None,
        degrees: float = 10.0,
        translate: float = 0.1,
        mosaic_scale: Tuple = (0.5, 1.5),
        mixup_scale: Tuple = (0.5, 1.5),
        shear: float = 2.0,
        enable_mixup: bool = True,
        mosaic_prob: float = 1.0,
        mixup_prob: float = 1.0,
        *args
    ):
        super().__init__(img_size, mosaic=mosaic)
        self._dataset = dataset
        self.preproc = preproc
        self.degrees = degrees
        self.translate = translate
        self.scale = mosaic_scale
        self.shear = shear
        self.mixup_scale = mixup_scale
        self.enable_mosaic = mosaic
        self.enable_mixup = enable_mixup
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.local_rank = 0

    def __len__(self):
        return len(self._dataset)

    @Dataset.mosaic_getitem
    def __getitem__(self, idx):
        if self.enable_mosaic and random.random() < self.mosaic_prob:
            mosaic_labels = []
            input_dim = self._dataset.input_dim
            input_h, input_w = input_dim[0], input_dim[1]

            yc = int(random.randint(0.5 * input_h, 1.5 * input_h))
            xc = int(random.randint(0.5 * input_w, 1.5 * input_w))

            indices = [idx] * [random.randint(0, len(self._dataset) - 1) for _ in range(3)]

            for i_mosaic, index in enumerate(indices):
                img, _labels, _, img_id = self._dataset.pull_item(index)
                h0, w0 = img.shape[:2] 

