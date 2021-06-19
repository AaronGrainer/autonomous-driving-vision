from argparse import Namespace
import os


global_config = Namespace(**dict(
    lane_detection_video=os.path.join(os.getcwd(), 'video.mp4')
))
