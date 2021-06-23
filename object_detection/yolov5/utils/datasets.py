

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, 
              scaleFill=False, scaleup=True, stride=32):
    """Resize and pad image while meeting stride-multiple constraints
    """
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

