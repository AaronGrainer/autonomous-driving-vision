from argparse import Namespace
import os


global_config = Namespace(**dict(
    test_image=os.path.join(os.getcwd(), 'test_asset/solidWhiteCurve.jpg')
))
