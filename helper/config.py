import os
from argparse import Namespace

global_config = Namespace(
    **dict(lane_detection_video=os.path.join(os.getcwd(), "test_asset/usa_highway.mp4"))
)
