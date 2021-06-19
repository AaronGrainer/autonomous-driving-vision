from argparse import Namespace
import os


global_config = Namespace(**dict(
    # Data
    data_root=os.path.join(os.getcwd(), 'lane_detection/datasets/'),
    list_path=os.path.join(os.getcwd(), 'lane_detection/datasets/list/test_split/test0_normal.txt'),
    output_dir=os.path.join(os.getcwd(), 'lane_detection/output'),

    # Network
    use_aux=True,
    griding_num=200,
    backbone='18',

    # Test
    test_model=os.path.join(os.getcwd(), 'lane_detection/checkpoint/culane_18.pth'),
    test_work_dir=os.path.join(os.getcwd(), 'lane_detection/test_output'),

    test_img=os.path.join(os.getcwd(), 'lane_detection/datasets/00810.jpg'),

    num_lanes=4
))
