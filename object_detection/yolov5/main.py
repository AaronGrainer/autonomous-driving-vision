import torch

from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances, Boxes

import cv2


COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
                'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 
                'teddy bear', 'hair drier', 'toothbrush']


def main():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    # img = 'https://ultralytics.com/images/zidane.jpg'
    img = 'test_asset/solidWhiteCurve.jpg'

    results = model(img)
    print('results: ', dir(results))

    print('results: ', results.pandas())

    MetadataCatalog.get(f"yolov5").set(thing_classes=COCO_CLASSES)
    yolov5_metadata = MetadataCatalog.get("yolov5")

    img = cv2.imread(img)
    height, width, _ = img.shape

    v = Visualizer(img[:, :, ::-1],
                   metadata=yolov5_metadata,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    # instance = Instances(
    #     image_size=(height, width),
    #     num_instances=80,
    #     fields=[{
    #         'pred_boxes': Boxes(),
    #         'scores': '',
    #         'pred_classes': ''
    #     }]
    # )
    # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2.imshow('Ballon Prediction', out.get_image()[:, :, ::-1])
    # cv2.waitKey()


if __name__ == '__main__':
    main()

