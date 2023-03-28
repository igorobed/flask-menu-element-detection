import numpy as np
from PIL import Image
import cv2
import logging
from skimage.metrics import structural_similarity

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


def show_img(img: np.ndarray) -> None:
    cv2.imshow("img", img)
    cv2.waitKey(0)


def custom_crop(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    return img[y : y + h, x : x + w]


def crop_head(img: np.ndarray, cropped_x_y) -> np.ndarray:
    """
    Обрезаем верхушку с временем, уровнем заряда,показателями телефона
    и полем с адресом страницы
    """
    cropped_x_y["y"] += 140
    return img[140:, :]


def get_changed_region(img_in: np.ndarray, img_out: np.ndarray):
    img_in_gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    img_out_gray = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)

    (score, diff) = structural_similarity(img_in_gray, img_out_gray, full=True)

    # если выполняется условие ниже, то считаем, что клик по бургер-меню не получился
    if (1 - score) < 0.005:
        return None, score * 100

    diff = (diff * 255).astype("uint8")
    diff_box = cv2.merge([diff, diff, diff])

    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    # найдем контур максимального размера
    max_c_size = 0
    max_c = None
    for c in contours:
        area = cv2.contourArea(c)
        if area > max_c_size:
            max_c_size = area
            max_c = c

    if type(max_c) != type(None):
        x, y, w, h = cv2.boundingRect(max_c)
        return {
            "x": x,
            "y": y,
            "w": w,
            "h": h,
        }, score * 100

    return {}, score * 100
