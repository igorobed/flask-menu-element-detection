# работаем с текстом
# предпочтение отдается тексту внутри drawer
# иерархическая кластеризация текста
# параллельные тексты исключаем

# преобладание элементов с одинаковым размером?
# как сделать серую область еще более серой?

# если есть тексты лежащие друг под другом и все, то кластеризация не нужна

# отслеживать перемешивание элементов кластера и в таком слкчае объединять их в один





from cvu.detector.yolov5 import Yolov5
from cvu.detector import Detector
from cvu.utils.draw import draw_bbox
import numpy as np
from PIL import Image
import cv2
import easyocr
import numpy as np
import time
from skimage.metrics import structural_similarity


classes = [
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


model = Yolov5(classes=classes,backend="torch",weight="detect_models/best25.torchscript",device="cpu",input_shape=640,)


def show_img(img: np.ndarray) -> None:
    cv2.imshow("img", img)
    cv2.waitKey(0)


def custom_crop(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    return img[y : y + h, x : x + w]


def crop_head(img: np.ndarray) -> np.ndarray:
    """
    Обрезаем верхушку с временем, уровнем заряда,показателями телефона
    и полем с адресом страницы
    """
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



inp_img_path = "tests\\test_imgs\\img_6_in.jpg"
out_img_path = "tests\\test_imgs\\img_6_out.jpg"

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

# menu_box_img = cv2.imread(out_img_path)

show_img(menu_box_img)

predictions = model(menu_box_img)

print(predictions)

for pred in predictions:
    print(pred.bbox)
    draw_bbox(menu_box_img, bbox=pred.bbox, title=pred.class_name)
    
show_img(menu_box_img)

