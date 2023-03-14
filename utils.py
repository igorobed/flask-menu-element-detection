import numpy as np
from PIL import Image
import cv2
import logging

my_logger = logging.getLogger(__name__)
my_logger.setLevel(logging.INFO)
handler = logging.FileHandler("logs.log", mode="w")
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
my_logger.addHandler(handler)


def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
