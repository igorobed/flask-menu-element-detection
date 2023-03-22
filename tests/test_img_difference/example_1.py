import numpy as np
import cv2


inp_img_path = "tests\\test_img_difference\\5e0f6568-422c-405f-917b-b56f0e5bb2c5.jpg"
out_img_path = "tests\\test_img_difference\\56b6b921-aa57-4ccb-8239-03f6c72179a1.jpg"


def crop_head(img: np.ndarray) -> np.ndarray:
    """
    Обрезаем верхушку с временем, уровнем заряда,показателями телефона
    и полем с адресом страницы
    """
    return img[140:, :]


def show_img(img: np.ndarray) -> None:
    cv2.imshow("img", img)
    cv2.waitKey(0)


img_in = cv2.imread(inp_img_path)
img_out = cv2.imread(out_img_path)

img_in = crop_head(img_in)
img_out = crop_head(img_out)

difference = cv2.subtract(img_in, img_in)

result = not np.any(difference)

if result:
    print("Same img")
else:
    show_img(difference)
    print("We have difference")
