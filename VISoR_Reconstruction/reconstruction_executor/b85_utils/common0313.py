import cv2
import numpy as np
import SimpleITK as sitk


def Preprocess(surface, threshold):
    surface = sitk.Threshold(surface, threshold, 65535, threshold)
    back_log_value = np.log(threshold)
    return sitk.Clamp(
        (sitk.Log(sitk.Cast(surface + 1, sitk.sitkFloat32)) - back_log_value) * 39.4,
        sitk.sitkUInt8,
        0,
        255,
    )


def fill_outside_yq(img, value: int):
    img[0, 0] = 0
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8)
    cv2.floodFill(img, mask, (0, 0), value, value, value, cv2.FLOODFILL_FIXED_RANGE)
    img[img.shape[0] - 1, 0] = 0
    cv2.floodFill(img, mask, (0, img.shape[0] - 1), value, value, value, cv2.FLOODFILL_FIXED_RANGE)
    img[img.shape[0] - 1, img.shape[1] - 1] = 0
    cv2.floodFill(
        img,
        mask,
        (img.shape[1] - 1, img.shape[0] - 1),
        value,
        value,
        value,
        cv2.FLOODFILL_FIXED_RANGE,
    )
    img[0, img.shape[1] - 1] = 0
    cv2.floodFill(img, mask, (img.shape[1] - 1, 0), value, value, value, cv2.FLOODFILL_FIXED_RANGE)
    return img
