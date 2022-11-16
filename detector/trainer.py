from pathlib import Path

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule, Trainer

from .models.yolox import YOLOX, YOLOPAFPN, YOLOXHead

from common import config


class YoloxModel(LightningModule):
    def __init__(self):
        self.warmup_epochs = config.YOLOX_CONFIG["warmup_epochs"]

    def forward(self):
        pass

    def training_step(self):
        pass

    def get_model(self):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(
                config.YOLOX_CONFIG["depth"],
                config.YOLOX_CONFIG["width"],
                in_channels=in_channels,
                act=config.YOLOX_CONFIG["act"]
            )
            head = YOLOPAFPN(
                config.YOLOX_CONFIG["num_classes"],
                config.YOLOX_CONFIG["width"],
                in_channels=in_channels,
                act=config.YOLOX_CONFIG["act"]
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()

        return self.model

    def configure_optimizers(self):
        warmup_lr = config.YOLOX_CONFIG["warmup_lr"]
        lr = config.YOLOX_CONFIG["basic_lr_per_img"] * config.YOLOX_CONFIG["batch_size"]

        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

        for k, v in self.model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)  # no decay
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay

        optimizer_warmup = torch.optim.SGD(
            pg0, lr=warmup_lr, momentum=config.YOLOX_CONFIG["momentum"], nesterov=True
        )
        optimizer_warmup.add_param_group(
            {"params": pg1, "weight_decay": config.YOLOX_CONFIG["weight_decay"]}
        )
        optimizer_warmup.add_param_group({"params": pg2})

        optimizer = torch.optim.SGD(
            pg0, lr=lr, momentum=config.YOLOX_CONFIG["momentum"], nesterov=True
        )
        optimizer.add_param_group(
            {"params": pg1, "weight_decay": config.YOLOX_CONFIG["weight_decay"]}
        )
        optimizer.add_param_group({"params": pg2})

        return [
            {"optimizer": optimizer_warmup, "frequency": config.YOLOX_CONFIG["warmup_epochs"]},
            {"optimizer": optimizer},
        ]


def train():
    # Initialize model
    yolox_model = YoloxModel()
    print('yolox_model: ', yolox_model)

    # # Resume model if enabled and checkpoint exists
    # if config.YOLOX_CONFIG["resume"]:
    #     checkpoint_file = Path(config.YOLOX_CONFIG["checkpoint_dir"], "checkpoint.ckpt")
    #     if checkpoint_file.is_file():
    #         yolox_model.load_from_checkpoint(checkpoint_file)

    # # Initialize dataloader

    # # Initialize trainer
    # trainer = Trainer(
    #     accelerator="auto",
    #     devices=1 if torch.cuda.is_available() else None,
    #     max_epochs=config.YOLOX_CONFIG["max_epoch"],
    #     default_root_dir=config.YOLOX_CONFIG["checkpoint_dir"],
    # )

    # # Train the model
    # # TODO: Add dataloader params
    # trainer.fit(yolox_model)
