from typing import Union

import cv2

from config import reader
from catboost import CatBoostRegressor
import numpy as np
from utils.global_utils import get_image_orientation, match_pic
import os
from utils.base_parser import BaseParser


class TgStatImageParser(BaseParser):
    MODEL_PATH = "ml_models/tg_model"
    MASK_DESKTOP_PATH = "masks/tg/tgstat/desktop"
    MASK_MOBILE_PATH = "masks/tg/tgstat/mobile"

    def __init__(self, img: np.ndarray):
        super().__init__(img)

    def check_image(self, cropped: np.ndarray) -> tuple[str, bool]:
        text = reader.readtext(cropped, detail=0, paragraph=True)
        ans, flag = "0", False
        if "err" in " ".join(text).lower():
            if len(text) == 3:
                ans = text[1]
                flag = True
            elif len(text) == 2:
                if "er" not in text[0].lower():
                    ans = text[0]
                    flag = True
        del cropped
        return ans, flag
