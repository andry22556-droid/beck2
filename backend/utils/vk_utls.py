import os
from typing import Union

import cv2

from config import reader
from utils.base_parser import BaseParser
import numpy as np

from utils.global_utils import get_image_orientation, match_pic
import re


class VkParser(BaseParser):
    MODEL_PATH = "ml_models/vk_model"
    MASK_DESKTOP_PATH = "masks/vk/desktop"
    MASK_MOBILE_PATH = "masks/vk/mobile"
    ANCHOR_WORDS = ["новые подписчики", "подписчики", "подписчиков", "падписчине", "участники", "подпнсчики"]

    def __init__(self, img: np.ndarray):
        super().__init__(img)

    @staticmethod
    def parse_text(text: list):
        ans, flag = "0", False
        pattern = r'\b\d+\b'
        matches = re.findall(pattern, " ".join(text).lower())
        if len(matches) == 1:
            int(matches[0])
            ans, flag = matches[0], True
        elif len(matches) > 1:
            for word in VkParser.ANCHOR_WORDS:
                for txt in reversed(text):
                    if word in txt:
                        matches2 = re.findall(pattern, txt)
                        if len(matches2) > 0:
                            ans, flag = matches2[0], True
                            break
                    if flag:
                        break
                if flag:
                    break
        return ans, flag

    def check_image(self, cropped: np.ndarray) -> tuple[str, bool]:
        text = reader.readtext(cropped, detail=0, paragraph=True)
        ans, flag = "0", False
        if len(text) <= 5:
            for word in self.ANCHOR_WORDS:
                if word in " ".join(text).lower():
                    ans, flag = self.parse_text(text)
                    break
        return ans, flag


