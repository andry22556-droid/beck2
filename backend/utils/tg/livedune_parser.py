import os
from typing import Union

import cv2

from config import reader
from utils.base_parser import BaseParser
import numpy as np
import re

from utils.global_utils import get_image_orientation, match_pic


class LiveDuneParser(BaseParser):
    MODEL_PATH = "ml_models/tg_model"
    MASK_DESKTOP_PATH = "masks/tg/livedune/desktop"
    MASK_MOBILE_PATH = "masks/tg/livedune/mobile"
    ANCHOR_WORDS = ["vr", "просмотр постов (ср", "просмотр постов"]

    def __init__(self, img: np.ndarray):
        super().__init__(img)

    @staticmethod
    def parse_text(text):
        pattern = r'(\d+\.\d+\s*%)|(\d+\s*%)'
        matches = re.findall(pattern, text)
        ans, flag = "0", False
        if len(matches) == 0:
            return "0", False
        for match in matches:
            try:
                val = match.replace("%", "")
                val = float(val)
                if val > 400:
                    continue
                ans = match
                flag = True
                break
            except:
                continue
        return ans, flag

    def check_image(self, cropped: np.ndarray) -> tuple[str, bool]:
        text = reader.readtext(cropped, detail=0, paragraph=True)
        ans, flag = "0", False
        for word in self.ANCHOR_WORDS:
            if word in " ".join(text).lower():
                for txt in reversed(text):
                    if "%" in txt:
                        ans, flag = self.parse_text(txt.lower())
                        if flag:
                            break
            if flag:
                break
        return ans, flag
