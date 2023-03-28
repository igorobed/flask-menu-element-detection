from cvu.detector.yolov5 import Yolov5
from cvu.utils.draw import draw_bbox
import numpy as np
from PIL import Image
from utils import (
    my_logger,
    convert_from_cv2_to_image,
    convert_from_image_to_cv2,
)


CLASSES_UI = [
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


class MyDetector:
    def __init__(self) -> None:
        self.model = Yolov5(
            classes="menu",
            backend="onnx",
            weight="detect_models/best.onnx",
            device="cpu",
            input_shape=640,
        )

    def __call__(self, img: Image, get_coords: bool = False) -> np.ndarray:
        img = convert_from_image_to_cv2(img)
        preds = self.model(img)
        my_logger.info("The model has processed the incoming image")

        # если алгоритм не обнаружил меню на изображении,
        # то просто возвращаем исходный объект

        found_elements = {}

        if len(preds) > 0:
            # модель может найти несколько элементов похожих на меню
            # необходимо выбрать наиболее вероятный
            max_conf = max([item.confidence for item in preds])
            for pred in preds:
                if pred.confidence == max_conf:
                    best_pred = pred
                    break

            # отобразим область с обнаруженным объектом
            # но сначала проверим, что найденный элемент не распологается снизу
            # иначе у нас произошло ложноположительное срабатывание
            if best_pred.bbox[1] < 2 * (img.shape[0] / 3):
                found_elements[best_pred.class_name] = {
                    "top_left": best_pred.bbox[:2],
                    "bottom_right": best_pred.bbox[2:],
                }
                draw_bbox(img, best_pred.bbox, color=(0, 0, 255))
        else:
            my_logger.info("The model did not find the desired object")

        img = convert_from_cv2_to_image(img)

        if not get_coords:
            return img
        else:
            return img, found_elements


class MyDetectorUI:
    def __init__(self) -> None:
        self.model = Yolov5(
        classes=CLASSES_UI,
        backend="torch",
        weight="detect_models/best25.torchscript",
        device="cpu",
        input_shape=640,
    )
        
    def __call__(self, img: Image) -> list:
        pass