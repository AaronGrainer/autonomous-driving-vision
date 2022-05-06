# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
General utils
"""

import contextlib
import glob
import logging
import math
import os
import platform
import random
import re
import signal
import time
import urllib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pkg_resources as pkg
import torch
import torchvision
import yaml

import src.vision.utils.dependency as _dependency

from .metrics import box_iou, fitness

# Settings
torch.set_printoptions(linewidth=320, precision=5, profile="long")
# format short g, %precision=5
np.set_printoptions(linewidth=320, formatter={"float_kind": "{:11.5g}".format})
pd.options.display.max_columns = 10
# prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
cv2.setNumThreads(0)
os.environ["NUMEXPR_MAX_THREADS"] = str(min(os.cpu_count(), 8))  # NumExpr max threads

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory


def set_logging(name=None, verbose=True):
    # Sets level and returns logger
    rank = int(os.getenv("RANK", -1))  # rank in world for Multi-GPU trainings
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if (verbose and rank in (-1, 0)) else logging.WARNING,
    )
    return logging.getLogger(name)


# Define globally (used in train.py, val.py, detect.py, etc.)
LOGGER = set_logging(__name__)


class Profile(contextlib.ContextDecorator):
    # Usage: @Profile() decorator or 'with Profile():' context manager
    def __enter__(self):
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        print(f"Profile results: {time.time() - self.start:.5f}s")


class Timeout(contextlib.ContextDecorator):
    # Usage: @Timeout(seconds) decorator or 'with Timeout(seconds):' context manager
    def __init__(self, seconds, *, timeout_msg="", suppress_timeout_errors=True):
        self.seconds = int(seconds)
        self.timeout_message = timeout_msg
        self.suppress = bool(suppress_timeout_errors)

    def _timeout_handler(self, signum, frame):
        raise TimeoutError(self.timeout_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self._timeout_handler)  # Set handler for SIGALRM
        signal.alarm(self.seconds)  # start countdown for SIGALRM to be raised

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)  # Cancel SIGALRM if it's scheduled
        if self.suppress and exc_type is TimeoutError:  # Suppress TimeoutError
            return True


def try_except(func):
    # try-except function. Usage: @try_except decorator
    def handler(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(e)

    return handler


def methods(instance):
    # Get class/instance methods
    return [f for f in dir(instance) if callable(getattr(instance, f)) and not f.startswith("__")]


def print_args(name, opt):
    # Print argparser arguments
    LOGGER.info(colorstr(f"{name}: ") + ", ".join(f"{k}={v}" for k, v in vars(opt).items()))


def init_seeds(seed=0):
    """Initialize random number generator (RNG) seeds
    https://pytorch.org/docs/stable/notes/randomness.html
    cudnn seed 0 settings are slower and more reproducible,
    else faster and less reproducible
    """
    import torch.backends.cudnn as cudnn

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)
