import cv2
import numpy as np
import skimage.exposure as exposure
import os
import fire


def get_trafic_light_color(img):
    # Calculate 2D histograms for pairs of channels: GR
    histGR = cv2.calcHist([img], [1, 2], None, [256, 256], [0, 256, 0, 256])

    # Histogram is float and counts need to be scale to range 0 to 255
    histScaled = exposure.rescale_intensity(histGR, in_range=(0,1), out_range=(0,255)).clip(0,255).astype(np.uint8)

    # Masks
    ww = 256
    hh = 256
    ww13 = ww // 3
    ww23 = 2 * ww13
    hh13 = hh // 3
    hh23 = 2 * hh13
    black = np.zeros_like(histScaled, dtype=np.uint8)
    # Specify points in OpenCV x,y format
    ptsUR = np.array( [[[ww13,0],[ww-1,hh23],[ww-1,0]]], dtype=np.int32 )
    redMask = black.copy()
    cv2.fillPoly(redMask, ptsUR, (255,255,255))
    ptsBL = np.array( [[[0,hh13],[ww23,hh-1],[0,hh-1]]], dtype=np.int32 )
    greenMask = black.copy()
    cv2.fillPoly(greenMask, ptsBL, (255,255,255))

    # Test histogram against masks
    region = cv2.bitwise_and(histScaled,histScaled,mask=redMask)
    redCount = np.count_nonzero(region)
    region = cv2.bitwise_and(histScaled,histScaled,mask=greenMask)
    greenCount = np.count_nonzero(region)
    # print('redCount:', redCount)
    # print('greenCount:', greenCount)

    # Find color
    threshCount = 40
    if redCount > greenCount and redCount > threshCount:
        color = "red"
    elif greenCount > redCount and greenCount > threshCount:
        color = "green"
    # elif redCount < threshCount and greenCount < threshCount:
    #     color = "yellow"
    else:
        color = "other"

    return color


def detect_trafic_light_color(img, preds):
    traffic_colors = []
    for _, pred in preds.iterrows():
        if pred['name'] == 'traffic light':
            traffic_img = img[int(pred['ymin']):int(pred['ymax']), int(pred['xmin']):int(pred['xmax'])]
            traffic_color = get_trafic_light_color(traffic_img[:, :, ::-1])
            traffic_colors.append(traffic_color)
        else:
            traffic_colors.append(0)
    preds['traffic_color'] = traffic_colors
    
    return preds

