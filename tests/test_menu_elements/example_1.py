# 1. Обрезать верхушку изображений
# 2. Получить иннфу о налиичии разности между ними
# 3. Получить выделенную контуром область
# 4. Задетектить в этой области текст
# 5. Вытащить текст, который относится

import cv2
import easyocr
import numpy as np
import time
from skimage.metrics import structural_similarity
from copy import deepcopy

# кластеризация ... ???
# from sklearn.cluster import AgglomerativeClustering


# class Point:
#     def __init__(self, x: int, y: int) -> None:
#         self.x = x
#         self.y = y


# class Box:
#     def __init__(self, )


class SimpleGroupDetector:
    def __init__(self, std: int = 7) -> None:
        self.std = std

    def __call__(self, boxes):
        # найти группы боксов, врехняя левая граница, которых по оси x
        # попадает в один диапазон
        boxes_range = {}
        for box in boxes:
            added_box = False
            # tl = (int(tl[0]), int(tl[1]))
            (tl, _, _, _) = box
            x = int(tl[0])
            if boxes_range == {}:
                boxes_range[self.get_key_range(x)] = [box]
                added_box = True
                continue

            for key in boxes_range.keys():
                if self.in_key_range(key, x):
                    boxes_range[key].append(box)
                    added_box = True
                    break

            if added_box:
                continue

            boxes_range[self.get_key_range(x)] = [box]
            added_box = True

        return boxes_range

    def get_key_range(self, x: int):
        min_border = 0
        if (x - self.std) >= 0:
            min_border = x - self.std
        max_border = x + self.std

        return f"{min_border}_{max_border}"

    def in_key_range(self, key_str: str, x: int):
        (min_border, max_border) = list(map(int, key_str.split("_")))
        bool_res = False
        if x <= max_border and x >= min_border:
            bool_res = True
        return bool_res


def crop_head(img: np.ndarray) -> np.ndarray:
    """
    Обрезаем верхушку с временем, уровнем заряда,показателями телефона
    и полем с адресом страницы
    """
    return img[140:, :]


def custom_crop(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    return img[y : y + h, x : x + w]


def show_img(img: np.ndarray) -> None:
    cv2.imshow("img", img)
    cv2.waitKey(0)


def get_menu_box(img_in: np.ndarray, img_out: np.ndarray):
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


# получаем два изображения
inp_img_path = "tests\\test_imgs\\img_5_in.jpg"
out_img_path = "tests\\test_imgs\\img_5_out.jpg"

img_in = cv2.imread(inp_img_path)
img_out = cv2.imread(out_img_path)

img_in = crop_head(img_in)
img_out = crop_head(img_out)

menu_box, score = get_menu_box(img_in, img_out)

# score выдает процент схожести двух изображений
if menu_box is None:
    print(f"Same image: {score}")
elif menu_box == {}:
    print("Menu field not detected")
else:
    print(f"Image changed: {score}")

# если я подам на вход не все изображение, а вырезанную область
# работа детектора текста ускорится?

# show_img(custom_crop(img_out, **menu_box))
menu_box_img = custom_crop(img_out, **menu_box)

# достанем текст внутри

start = time.time()

reader = easyocr.Reader(["ru", "en"])
results = reader.readtext(menu_box_img, detail=1, paragraph=False)
# print(results)

end = time.time()

for bbox, text, prob in results:
    (tl, tr, br, bl) = bbox
    tl = (int(tl[0]), int(tl[1]))
    tr = (int(tr[0]), int(tr[1]))
    br = (int(br[0]), int(br[1]))
    bl = (int(bl[0]), int(bl[1]))

    cv2.rectangle(menu_box_img, tl, br, (0, 255, 0), 2)

# show_img(menu_box_img)

# сейчас я предварительно буду формировать список из боксов
# и уже потом формировать словарь диапазонов
# но после делать формирование словаря диапазонов налету

detector = SimpleGroupDetector()
list_bboxes = []
for bbox, text, prob in results:
    list_bboxes.append(bbox)

res = detector(list_bboxes)

# нарисуем найденные области
# для каждой области нужно отобрать самый высокий элемент и самый низкий
union_rectangles = []
for key, items in res.items():
    (tl, tr, br, bl) = items[0]
    tl = (int(tl[0]), int(tl[1]))
    br = (int(br[0]), int(br[1]))
    if len(items) == 1:
        union_rectangles.append([tl, br])
        continue

    top_left_x_y, back_right_x_y = tl, br

    for box in items[1:]:
        (tl, tr, br, bl) = box
        tl = (int(tl[0]), int(tl[1]))
        br = (int(br[0]), int(br[1]))

        if tl[1] < top_left_x_y[1]:
            top_left_x_y = tl

        if br[1] > back_right_x_y[1]:
            back_right_x_y = br

    union_rectangles.append([top_left_x_y, back_right_x_y])

# отрисовываем найденные области
for item in union_rectangles:
    cv2.rectangle(menu_box_img, item[0], item[1], (0, 0, 255), 2)

show_img(menu_box_img)

cv2.imwrite("tests\\test_menu_elements\\img_5_menu.jpg", menu_box_img)

print(f"{int((end - start) * 1000)} ms")
