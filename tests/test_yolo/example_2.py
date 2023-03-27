from cvu.detector.yolov5 import Yolov5
from cvu.utils.draw import draw_bbox
import numpy as np
import cv2
import numpy as np
from skimage.metrics import structural_similarity
import time
from sklearn.cluster import AgglomerativeClustering
from copy import deepcopy


CLASSES = [
    'BackgroundImage',
    'Bottom_Navigation',
    'Card',
    'CheckBox',
    'Checkbox',
    'CheckedTextView',
    'Drawer',
    'EditText',
    'Icon',
    'Image',
    'Map',
    'Modal',
    'Multi_Tab',
    'PageIndicator',
    'Remember',
    'Spinner',
    'Switch',
    'Text',
    'TextButton',
    'Toolbar',
    'UpperTaskBar'
    ]


model = Yolov5(
    classes=CLASSES,
    backend="torch",
    weight="detect_models/best25.torchscript",
    device="cpu",
    input_shape=640,
    )


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


def search_drawer(predictions):
    for pred in predictions:
        if pred.class_name == "Drawer":
            return pred.bbox
    return None


if __name__ == "__main__":
    # получим изображения
    inp_img_path = "tests\\test_imgs\\img_6_in.jpg"
    out_img_path = "tests\\test_imgs\\img_6_out.jpg"

    img_in = cv2.imread(inp_img_path)
    img_out = cv2.imread(out_img_path)

    cropped_x_y_in = {
        "x": 0,
        "y": 0
    }

    cropped_x_y_out = {
        "x": 0,
        "y": 0
    }

    img_in = crop_head(img_in, cropped_x_y=cropped_x_y_in)
    img_out = crop_head(img_out, cropped_x_y=cropped_x_y_out)

    menu_box, score = get_menu_box(img_in, img_out)

    # score выдает процент схожести двух изображений
    if menu_box is None:
        print(f"Same image: {score}")
    elif menu_box == {}:
        print("Menu field not detected")
    else:
        print(f"Image changed: {score}")

    # вырезаем область, притерпевшую изменения
    menu_box_img = custom_crop(img_out, **menu_box)

    start = time.time()
    predictions_out = model(img_out)
    predictions_crop = model(menu_box_img)
    end = time.time()
    print(end - start)

    # ищем, есть ли Drawer в предсказаниях
    drawer_bbox = None
    drawer_coords = search_drawer(predictions_out)
    drawer_coords_crop = search_drawer(predictions_crop)

    if drawer_coords_crop:
        drawer_bbox = drawer_coords_crop
    elif drawer_coords:
        drawer_bbox = drawer_coords
    
    # искать кластеры текстов мне в любом случае надо в обрезанном menu_box_img
    clustering = AgglomerativeClustering(
	n_clusters=2,
	affinity="manhattan",
	linkage="complete",
    )

    list_bboxes = []
    list_bboxes_tl = []

    for item in predictions_crop:
        if item.class_name == "Text":
            list_bboxes.append(list(map(int, item.bbox)))
            list_bboxes_tl.append(list(map(int, item.bbox[:2])))

    for item in list_bboxes_tl:
        item[1] = 0

    clustering.fit(list_bboxes_tl)

    # теперь мне надо найти области и их площади, чтобы выбрать максимальную
    unions = {}
    for idx, label in enumerate(clustering.labels_):
        if label in unions:
            unions[label].append(list_bboxes[idx])
        else:
            unions[label] = [list_bboxes[idx]]

    img_copy = deepcopy(menu_box_img)

    for cl, bboxes in unions.items():
    
        if cl == 0:
            for bbox in bboxes:
                tl, br = bbox[:2], bbox[2:]
                cv2.rectangle(img_copy, tl, br, (255, 0, 0), 2)
        if cl == 1:
            for bbox in bboxes:
                tl, br = bbox[:2], bbox[2:]
                cv2.rectangle(img_copy, tl, br, (0, 255, 0), 2)

    show_img(img_copy)

    union_rectangles = []
    for key, items in unions.items():
        tl, br = items[0][:2], items[0][2:]
    
        if len(items) == 1:
            union_rectangles.append([tl, br])
            continue

        top_left_x_y, back_right_x_y = tl, br

        min_x = top_left_x_y[0]
        max_x = back_right_x_y[0]

        for box in items[1:]:
            tl, br = box[:2], box[2:]

            if tl[1] < top_left_x_y[1]:
                top_left_x_y = tl

            if br[1] > back_right_x_y[1]:
                back_right_x_y = br

            if back_right_x_y[0] > max_x:
                max_x = back_right_x_y[0]
        
            if top_left_x_y[0] < min_x:
                min_x = top_left_x_y[0]

        top_left_x_y = [min_x, top_left_x_y[1]]
        back_right_x_y = [max_x, back_right_x_y[1]]
        union_rectangles.append([top_left_x_y, back_right_x_y])

    # for item_rect in union_rectangles:
    #     cv2.rectangle(img_copy, item_rect[0], item_rect[1], (0, 0, 255), 2)

    # show_img(img_copy)

    # проверием две области на пересечение
    # left = max(r1.left, r2.left)
    left = max(union_rectangles[0][0][0], union_rectangles[1][0][0])
    # right = min(r1.right, r2.right)
    right = min(union_rectangles[0][1][0], union_rectangles[1][1][0])
    # bottom = min(r1.bottom, r2.bottom)
    bottom = min(union_rectangles[0][1][1], union_rectangles[1][1][1])
    # top = max(r1.top, r2.top)
    top = max(union_rectangles[0][0][1], union_rectangles[1][0][1])

    # if left < right and bottom > top:
    #     print("Intersect")

    # https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc
    # xA = max(boxA[0], boxB[0]) max(union_rectangles[0][0][0], union_rectangles[1][0][0])
    # yA = max(boxA[1], boxB[1]) max(union_rectangles[0][0][1], union_rectangles[1][0][1])
    # xB = min(boxA[2], boxB[2]) min(union_rectangles[0][1][0], union_rectangles[1][1][0])
    # yB = min(boxA[3], boxB[3]) min(union_rectangles[0][1][1], union_rectangles[1][1][1])

    

    inter_area = abs(max((right - left, 0)) * max((bottom - top), 0))
    print(inter_area)

    union_rectangles_squares = []
    for item in union_rectangles:
        curr_square = (item[1][0] - item[0][0]) * (item[1][1] - item[0][1])
        union_rectangles_squares.appends(curr_square)
        print(curr_square)

    max_square = max(union_rectangles_squares[0], union_rectangles_squares[1])
    min_square = max(union_rectangles_squares[0], union_rectangles_squares[1])

    if (1 - (float(min_square) / max_square)) < 0.2:
        # тут считаем например, где текстов больше и если в мин текстов больше, то его делаем макс и принимаем за финальный
        pass
    

    max_square = 0
    max_rectangles = None
    union_rectangles_squares = []
    for item in union_rectangles:
        curr_square = (item[1][0] - item[0][0]) * (item[1][1] - item[0][1])
        print(curr_square)
        # union_rectangles_squares.appends(curr_square)
        if curr_square > max_square:
            max_rectangles = item
            max_square = curr_square
    
    # if inter_area > 0:
    #     if inter_area in union_rectangles_squares:
    #         max_square = max_square
    #     else:
            # посчитаем, какой процент площади каждой из областей занимает пересечение

    cv2.rectangle(img_copy, max_rectangles[0], max_rectangles[1], (0, 0, 255), 2)

    show_img(img_copy)

    # intersect


     # если две найденные области пересекаются или одна вложена в другую,то сливаем их
     # посмотреть на расположение элементов относительно друг друга внутри полученного бокса
     # в линию
     # сделать кластеризацию элементов по площади(а мб еще и лежащие на одной линии так же искать???)