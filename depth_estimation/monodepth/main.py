import os
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import fire

import torch
from torchvision import transforms, datasets

import depth_estimation.monodepth.networks
from depth_estimation.monodepth.layers import disp_to_depth
from depth_estimation.monodepth.utils import download_model_if_doesnt_exist
# from depth_estimation.monodepth.evaluate_depth import STEREO_SCALE_FACTOR


def main(detect_type='image'):
    if detect_type == 'image':
        img_path = 'test_asset/usa_laguna_moment.jpg'
    else:
        raise NotImplemented


if __name__ == '__main__':
    fire.Fire(main)

