import numpy as np
import re
from config import reader
from utils.base_parser import BaseParser


class ZnParser(BaseParser):
    MODEL_PATH = "ml_models/zn_model"
    MASK_DESKTOP_PATH = "masks/zn/desktop"
    MASK_MOBILE_PATH = "masks/zn/mobile"
    ANCHOR_WORDS = ['подписки', "количество дочитываний", "дачить ванив прасматрь"]

    def __init__(self, img: np.ndarray):
        super().__init__(img)

    @staticmethod
    def parse_text(text: list):
        pattern = r'(\d+\s+\d+)'
        ans, flag = "0", False
        matches = re.findall(pattern, " ".join(text).lower())
        if len(matches) == 1:
            try:
                matches[0] = matches[0].replace(" ", "")
                int(matches[0])
                ans, flag = matches[0], True
            except ValueError:
                pass
        elif len(matches) > 1:
            ans, flag = matches[1], True
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
