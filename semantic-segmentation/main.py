from json import load
import torch
import detectron2
import detectron2.utils.comm as comm
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_cityscapes_instances, builtin_meta
from detectron2.engine import DefaultPredictor, DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import CityscapesInstanceEvaluator
from detectron2.utils.events import get_event_storage
from detectron2.utils.visualizer import Visualizer

import glob
import os
import sys
import numpy as np
from fvcore.common.file_io import PathManager
from collections import OrderedDict
import fire


def register_dataset_instance(image_dir, gt_dir, splits=['train', 'val'], dataset_name='cityscapes', from_json=True):
    for split in splits:
        meta = builtin_meta._get_builtin_metadata('cityscapes')

        dataset_instance_name = f'{str(dataset_name)}_instance_{str(split)}'
        image_split_dir = os.path.join(image_dir, split)
        gt_split_dir = os.path.join(gt_dir, split)

        DatasetCatalog.register(dataset_instance_name,
                                lambda x=image_split_dir, y=gt_split_dir: load_cityscapes_instances(
                                    x, y, from_json=from_json, to_polygons=True
                                ))
        MetadataCatalog.get(dataset_instance_name).set(image_dir=image_split_dir,
                                                       gt_dir=gt_split_dir,
                                                       evaluator_type='cityscapes_instance',
                                                       **meta)
        print(f'Registered {dataset_instance_name} to DatasetCatalog.')



def main(do_eval=False):
    dataset_root_dir = 'datasets/kitti_semantics_cs'
    dataset_name = dataset_root_dir.split('/')[-1]
    image_dir = os.path.join(dataset_root_dir, "data_semantics/")
    gt_dir = os.path.join(dataset_root_dir, "gtFine/")

    # Register Dataset
    splits = ['train', 'val'] if do_eval else ['train']
    register_dataset_instance(image_dir, gt_dir, splits=splits, dataset_name=dataset_name, from_json=False)
    dataset_train = f'{dataset_name}_instance_train'
    dataset_val = f'{dataset_name}_instance_val'



if __name__ == '__main__':
    fire.Fire(main)

