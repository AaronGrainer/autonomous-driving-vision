from argparse import Namespace
import os


global_config = Namespace(**dict(
    # Data
    dataset='CULane',
    data_root=os.path.join(os.getcwd(), 'dataset/culane/'),

    # Network
    use_aux=True,
    griding_num=200,
    backbone='18',

    # Test
    test_model=os.path.join(os.getcwd(), 'checkpoint/culane_18.pth'),
    test_work_dir=os.path.join(os.getcwd(), 'test_output'),

    num_lanes=4
))
