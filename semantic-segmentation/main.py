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


def setup(do_eval, output_dir, dataset_train, dataset_val):
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file("Cityscapes/mask_rcnn_R_50_FPN.yaml"))
    cfg.OUTPUT_DIR = f'{output_dir}/output_resnet-50'
    cfg.DATASETS.TRAIN = (dataset_train,)
    cfg.DATASETS.TEST = (dataset_val,) if do_eval else ()

    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Cityscapes/mask_rcnn_R_50_FPN.yaml")
    cfg.TEST.EVAL_PERIOD = 600

    cfg.SOLVER.IMS_PER_BATCH = 16
    cfg.SOLVER.BASE_LR = 0.01
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.STEPS = (10000, 20000)  # iteration numbers to decrease learning rate by SOLVER.GAMMA
    cfg.SOLVER.MAX_ITER = 30000

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    return cfg


class SegmentationTrainer(DefaultTrainer):
    """Create a subclass inheriting from DefaultTrainer.
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """Create an evaluator for cityscapes_instance evaluation
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, 'validation')
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        assert evaluator_type == 'cityscapes_instance'
        assert(torch.cuda.device_count() >= comm.get_rank()), \
            'CityscapesEvaluator currently does not work with multiple machines.'

        return MyCityscapesInstanceEvaluator(dataset_name)


class MyCityscapesInstanceEvaluator(CityscapesInstanceEvaluator):
    def evaluate(self):
        """Overwrite the evaluate method in CityscapesInstanceEvaluator.
           Add lines to write AP scores to be visualized in Tensorboard.
        """
        comm.synchronize()
        if comm.get_rank() > 0:
            return
        import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling as cityscapes_eval

        self._logger.info(f'Evaluating results under {self._temp_dir} ...')

        cityscapes_eval.args.predictionPath = os.path.abspath(self._temp_dir)
        cityscapes_eval.args.predictionWalk = None
        cityscapes_eval.args.JSONOutput = False
        cityscapes_eval.args.colorized = False
        cityscapes_eval.args.gtInstancesFile = os.path.join(self._temp_dir, 'gtInstances.json')

        gt_dir = PathManager.get_local_path(self._metadata.gt_dir)
        groundTruthImgList = glob.glob(os.path.join(gt_dir, '*', '*_gtFine_instanceIds.png'))
        assert len(groundTruthImgList), \
            f'Cannot find any ground truth images to use for evaluation. Searched for: {cityscapes_eval.args.groundTruthSearch}'

        predictionImgList = []
        for gt in groundTruthImgList:
            predictionImgList.append(cityscapes_eval.getPrediction(gt, cityscapes_eval.args))
        results = cityscapes_eval.evaluateImgLists(predictionImgList,
                                                   groundTruthImgList,
                                                   cityscapes_eval.args)['averages']
        
        res = OrderedDict()
        res['segm'] = {
            'AP': results['allAp']*100,
            'AP50': results['allAp50%']*100
        }

        # Writing to tensorboard
        storage = get_event_storage()
        storage.put_scalar('eval/AP', res['segm']['AP'])
        storage.put_scalar('eval/AP50', res['segm']['AP50'])

        self._working_dir.cleanup()
        return res


def main(do_eval=False, output_dir='/mark_rcnn_output'):
    dataset_root_dir = 'datasets/kitti_semantics_cs'
    dataset_name = dataset_root_dir.split('/')[-1]
    image_dir = os.path.join(dataset_root_dir, "data_semantics/")
    gt_dir = os.path.join(dataset_root_dir, "gtFine/")

    # Register Dataset
    splits = ['train', 'val'] if do_eval else ['train']
    register_dataset_instance(image_dir, gt_dir, splits=splits, dataset_name=dataset_name, from_json=False)
    dataset_train = f'{dataset_name}_instance_train'
    dataset_val = f'{dataset_name}_instance_val'
    
    cfg = setup(do_eval, output_dir, dataset_train, dataset_val)

    trainer = SegmentationTrainer(cfg)
    trainer.resume_or_load(resume=False)
    return trainer.train()


if __name__ == '__main__':
    fire.Fire(main)

