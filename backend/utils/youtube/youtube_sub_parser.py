from typing import Union

import numpy as np
import re

from config import reader
from utils.base_parser import BaseParser
import random


class YoutubeSubscriberParser(BaseParser):
    MODEL_PATH = "ml_models/yt1_model"
    MASK_DESKTOP_PATH = "masks/yt1/desktop"
    MASK_MOBILE_PATH = "masks/yt1/mobile"
    ANCHOR_WORDS = ['пъдпьсчикl', "подпкечики", "подписчики", "поппьечину"]

    def __init__(self, img: np.ndarray):
        super().__init__(img)

    @staticmethod
    def parse_text(text: list):
        pattern = r'\b\d+\b'
        ans, flag = "0", False
        matches = re.findall(pattern, " ".join(text).lower())
        if len(matches) == 1:
            ans, flag = matches[0], True
        elif len(matches) > 1:
            idx = random.randint(0, len(matches) - 1)
            ans, flag = matches[idx], True
        return ans, flag

    def check_image(self, cropped: np.ndarray) -> tuple[str, bool]:
        text = reader.readtext(cropped, detail=0, paragraph=True)
        ans, flag = "0", False
        if len(text) <= 6:
            for word in self.ANCHOR_WORDS:
                if word in " ".join(text).lower():
                    ans, flag = self.parse_text(text)
        return ans, flag
