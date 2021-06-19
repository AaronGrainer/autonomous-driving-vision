import torch

import os
from PIL import Image


class LaneTestDataset(torch.utils.data.Dataset):
    def __init__(self, path, list_path, img_transform=None):
        super(LaneTestDataset, self).__init__()

        self.path = path
        self.img_transform = img_transform

        with open(list_path, 'r') as f:
            self.list = f.read().splitlines()
        
    def __getitem__(self, index):
        name = self.list[index]
        img_path = os.path.join(self.path, name)
        img = Image.open(img_path)

        if self.img_transform is not None:
            img = self.img_transform(img)

        return img, name

    def __len__(self):
        return len(self.list)

