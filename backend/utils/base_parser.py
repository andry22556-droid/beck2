import os
from abc import ABC, abstractmethod
from typing import Union

import cv2
import numpy as np
from catboost import CatBoostRegressor

from config import reader
from utils.global_utils import get_image_orientation, match_pic


class BaseParser(ABC):
    MODEL_PATH = None
    MASK_DESKTOP_PATH = None
    MASK_MOBILE_PATH = None
    ANCHOR_WORDS = None

    def __init__(self, img: np.ndarray):
        self._img = img
        self.model = CatBoostRegressor()
        self.model.load_model(self.MODEL_PATH)

    def _mobile_image_reader(self) -> Union[str, float, int]:
        ans, flag = 0, False
        for mobile_mask in os.listdir(self.MASK_MOBILE_PATH):
            template = cv2.resize(cv2.imread(f"{self.MASK_MOBILE_PATH}/{mobile_mask}", 0), (0, 0), fx=0.8, fy=0.8)
            for cropped in match_pic(self._img, template):
                ans, flag = self.check_image(cropped)
                if flag:
                    break
            if flag:
                break
        if not flag:
            ans = self.model.predict(" ".join(reader.readtext(self._img, detail=0, paragraph=True)))
        return ans

    def _desktop_image_reader(self) -> Union[str, float, int]:
        ans, flag = 0, False
        for desktop_mask in os.listdir(self.MASK_DESKTOP_PATH):
            template = cv2.resize(cv2.imread(f"{self.MASK_DESKTOP_PATH}/{desktop_mask}", 0), (0, 0), fx=0.8, fy=0.8)
            for cropped in match_pic(self._img, template):
                ans, flag = self.check_image(cropped)
                if flag:
                    break
            if flag:
                break
        if not flag:
            arr = np.array([" ".join(reader.readtext(self._img, detail=0, paragraph=True))])
            ans = self.model.predict(arr)
        return ans

    def run_pic(self) -> Union[str, float, int]:
        orientaion = get_image_orientation(self._img)
        if orientaion == "vertical":
            return self._mobile_image_reader()
        else:
            return self._desktop_image_reader()

    @abstractmethod
    def check_image(self, cropped: np.ndarray) -> tuple[str, bool]:
        raise NotImplementedError
