import logging
import math
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

import src.vision.utils.dependency as _dependency
from src.vision.v5.utils.autoanchor import check_anchor_order
from src.vision.v5.utils.general import make_divisible
from src.vision.v5.utils.torch_utils import (
    fuse_conv_and_bn,
    initialize_weights,
    model_info,
    scale_img,
    time_sync,
)
