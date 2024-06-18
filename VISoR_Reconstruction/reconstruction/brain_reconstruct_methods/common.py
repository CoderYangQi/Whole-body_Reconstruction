import SimpleITK as sitk
import cv2
import numpy as np


def _fill_outside(img, value):
    img[0, 0] = 0
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8)
    cv2.floodFill(img,
                  mask,
                  (0, 0), value, value, value, cv2.FLOODFILL_FIXED_RANGE)
    img[img.shape[0] - 1, 0] = 0
    cv2.floodFill(img,
                  mask,
                  (0, img.shape[0] - 1), value, value, value, cv2.FLOODFILL_FIXED_RANGE)
    img[img.shape[0] - 1, img.shape[1] - 1] = 0
    cv2.floodFill(img,
                  mask,
                  (img.shape[1] - 1, img.shape[0] - 1), value, value, value, cv2.FLOODFILL_FIXED_RANGE)
    img[0, img.shape[1] - 1] = 0
    cv2.floodFill(img,
                  mask,
                  (img.shape[1] - 1, 0), value, value, value, cv2.FLOODFILL_FIXED_RANGE)
    return img

def fill_outside(image: sitk.Image, value: int):
    image = sitk.GetArrayFromImage(image)
    if len(image.shape) == 2:
        image = _fill_outside(image, value)
    elif len(image.shape) == 3:
        for i in range(image.shape[0]):
            ch = image[i]
            _fill_outside(ch, value)
            image[i] = ch
    image = sitk.GetImageFromArray(image)
    return image
