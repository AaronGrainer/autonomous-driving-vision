import cv2
import numpy as np


def get_lane(prob_map, gap, ppl, thresh, resize_shape=None):
    """
    (Adapted from harryhan618/SCNN_Pytorch)
    Note that in tensors we have indices start from 0 and in annotations coordinates start at 1
    Args:
        prob_map: prob map for single lane, np array size (h, w)
        gap: y pixel gap for sampling
        ppl: how many points for one lane
        thresh: probability threshold
        resize_shape: reshape size target, (H, W)

    Raises:
        ValueError

    Returns:
        coords: x coords bottom up every gap px, 0 for non-exist, in resized shape
    """
    if resize_shape is None:
        resize_shape = prob_map.shape
    h, w = prob_map.shape
    H, W = resize_shape
    coords = np.zeros(ppl)
    for i in range(ppl):
        y = int(h - i * gap / H * h - 1)  # Same as original SCNN code
        if y < 0:
            break
        line = prob_map[y, :]
        id = np.argmax(line)
        if line[id] > thresh:
            coords[i] = int(id / w * W)
    if (coords > 0).sum() < 2:
        coords = np.zeros(ppl)
    return coords


def prob_to_lines(seg_pred, exist, resize_shape=None, smooth=True, gap=20, ppl=None, thresh=0.3):
    """
    (Adapted from harryhan618/SCNN_Pytorch)
    Args:
        seg_pred: np.array size (num_classes, h, w)
        exist: list of existence, e.g. [0, 1, 1, 0]
        resize_shape: reshape size target, (H, W)
        smoot: whether to smooth the probability or not
        gap: y pixel gap for sampling
        ppl: how many points for one lane
        thresh: probability threshold

    Returns:
        coordinates: [x, y] list of lanes, e.g.: [ [[9, 569], [50, 549]] ,[[630, 569], [647, 549]] ]
    """
    if resize_shape is None:
        resize_shape = seg_pred.shape[1:]  # seg_pred (num_classes, h, w)
    _, h, w = seg_pred.shape
    H, W = resize_shape
    coordinates = []

    if ppl is None:
        ppl = round(H / 2 / gap)

    for i in range(1, seg_pred.shape[0]):
        prob_map = seg_pred[i, :, :]
        if exist[i - 1]:
            if smooth:
                prob_map = cv2.blur(prob_map, (9, 9), borderType=cv2.BORDER_REPLICATE)
            coords = get_lane(prob_map, gap, ppl, thresh, resize_shape)
            if coords.sum() == 0:
                continue
            coordinates.append([[coords[j], H - j * gap - 1] for j in range(ppl) if coords[j] > 0])

    return coordinates
