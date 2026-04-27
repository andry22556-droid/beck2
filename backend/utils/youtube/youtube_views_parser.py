import numpy as np

from utils.base_parser import BaseParser


class YouTubeViewsParser(BaseParser):
    MODEL_PATH = "ml_models/yt2_model"
    MASK_DESKTOP_PATH = "masks/yt2/desktop"
    MASK_MOBILE_PATH = "masks/yt2/mobile"
    ANCHOR_WORDS = ['пъдпьсчикl', "подпкечики", "подписчики", "поппьечину"]

    def __init__(self, img: np.ndarray):
        super().__init__(img)

    def check_image(self, cropped: np.ndarray) -> tuple[str, bool]:
        raise NotImplementedError
