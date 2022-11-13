import torch.nn as nn

from .yolo_head import YOLOXHEAD
from .yolo_pafpn import YOLOPAFPN


class YOLOX(nn.Module):
    """YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHEAD(80)

