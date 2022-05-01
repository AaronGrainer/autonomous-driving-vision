import os
from argparse import Namespace

global_config = Namespace(
    **dict(
        # Data
        data_root=os.path.join(os.getcwd(), "lane_detection/datasets/"),
        list_path=os.path.join(
            os.getcwd(), "lane_detection/datasets/list/test_split/test0_normal.txt"
        ),
        output_dir=os.path.join(os.getcwd(), "lane_detection/output"),
        # Network
        use_aux=True,
        griding_num=200,
        backbone="18",
        # Test
        test_model=os.path.join(os.getcwd(), "lane_detection/checkpoint/culane_18.pth"),
        test_work_dir=os.path.join(os.getcwd(), "lane_detection/test_output"),
        # test_img=os.path.join(os.getcwd(), 'lane_detection/datasets/05250523_0308.MP4/00870.jpg'),
        test_img=os.path.join(os.getcwd(), "test_asset/video_moment.jpg"),
        test_video=os.path.join(os.getcwd(), "test_asset/usa_highway.mp4"),
        num_lanes=4,
    )
)
