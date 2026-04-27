import os
import shutil

import cv2
import numpy as np


def get_image_orientation(image: np.ndarray):
    height, width = image.shape
    if height > width:
        return "vertical"
    else:
        return "horzontal"


def crop_pic(img, location, w, h):
    cropped_img = img[location[1]:location[1] + h, location[0]:location[0] + w]
    return cropped_img


def match_pic(img: np.ndarray, template: np.ndarray):
    methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
               cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
    h, w = template.shape
    for method in methods:
        img2 = img.copy()

        result = cv2.matchTemplate(img2, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            location = min_loc
        else:
            location = max_loc

        bottom_right = (location[0] + w, location[1] + h)
        cv2.rectangle(img2, location, bottom_right, 0, 5)
        cropped_img = crop_pic(img2, location, w, h)
        yield cropped_img


def clear_extract_dir():
    folder = "extract"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
