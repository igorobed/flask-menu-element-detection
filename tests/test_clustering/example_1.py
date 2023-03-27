# https://pyimagesearch.com/2022/02/28/multi-column-table-ocr/
import cv2
import easyocr
import numpy as np
import time
from skimage.metrics import structural_similarity
from sklearn.cluster import AgglomerativeClustering


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
inp_img_path = "tests\\test_imgs\\img_12_in.jpg"
out_img_path = "tests\\test_imgs\\img_12_out.jpg"

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

show_img(menu_box_img)

reader = easyocr.Reader(["ru", "en"])
results = reader.readtext(menu_box_img, detail=1, paragraph=False)

list_bboxes = []
list_bboxes_tl = []

for bbox, text, prob in results:
    (tl, tr, br, bl) = bbox
    tl = (int(tl[0]), int(tl[1]))
    tr = (int(tr[0]), int(tr[1]))
    br = (int(br[0]), int(br[1]))
    bl = (int(bl[0]), int(bl[1]))

    # cv2.rectangle(menu_box_img, tl, br, (0, 255, 0), 2)

    list_bboxes.append((tl, tr, br, bl))
    list_bboxes_tl.append([tl[0], 0])
    # list_bboxes_tl.append(tl)

# show_img(menu_box_img)

clustering = AgglomerativeClustering(
	n_clusters=2,
	affinity="manhattan",
	linkage="complete",
    )
clustering.fit(list_bboxes_tl)

print(clustering.labels_)

# for cl, bbox in zip(clustering.labels_, list_bboxes):
#     (tl, tr, br, bl) = bbox
#     if cl == 0:
#         cv2.rectangle(menu_box_img, tl, br, (255, 0, 0), 2)
#     if cl == 1:
#         cv2.rectangle(menu_box_img, tl, br, (0, 255, 0), 2)
#     if cl == 2:
#         cv2.rectangle(menu_box_img, tl, br, (0, 0, 255), 2)

unions = {}
for idx, label in enumerate(clustering.labels_):
    if label in unions:
        unions[label].append(list_bboxes[idx])
    else:
        unions[label] = [list_bboxes[idx]]

union_rectangles = []
for key, items in unions.items():
    (tl, tr, br, bl) = items[0]
    
    if len(items) == 1:
        union_rectangles.append([tl, br])
        continue

    top_left_x_y, back_right_x_y = tl, br

    max_x = back_right_x_y[0]

    for box in items[1:]:
        (tl, tr, br, bl) = box
        tl = (int(tl[0]), int(tl[1]))
        br = (int(br[0]), int(br[1]))

        if tl[1] < top_left_x_y[1]:
            top_left_x_y = tl

        if br[1] > back_right_x_y[1]:
            back_right_x_y = br

        if back_right_x_y[0] > max_x:
            max_x = back_right_x_y[0]

    back_right_x_y = (max_x, back_right_x_y[1])
    union_rectangles.append([top_left_x_y, back_right_x_y])

# for item in union_rectangles:
#     cv2.rectangle(menu_box_img, item[0], item[1], (0, 0, 255), 2)

max_square = 0
max_rectangles = None
for item in union_rectangles:
    curr_square = (item[1][0] - item[0][0]) * (item[1][1] - item[0][1])
    if curr_square > max_square:
        max_rectangles = item
        max_square = curr_square

cv2.rectangle(menu_box_img, max_rectangles[0], max_rectangles[1], (0, 0, 255), 2)

show_img(menu_box_img)


# предварительно обработать результаты детекции текста и слить несколько боксов в один если они слишком близко по оси X
# строки с несколькими элементами в одной строке выкидываются
