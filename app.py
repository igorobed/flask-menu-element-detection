from flask import Flask, render_template, request
from PIL import Image
from io import BytesIO
import base64
from utils import (
    convert_from_cv2_to_image,
    convert_from_image_to_cv2,
)
from detect import model
import time
from cvu.utils.draw import draw_bbox
import logging

app = Flask(__name__)
# logging.basicConfig(
#     level=logging.INFO,
#     filename="logs.log",
#     filemode="w",
#     format="%(asctime)s %(levelname)s %(message)s",
# )

my_logger = logging.getLogger(__name__)
my_logger.setLevel(logging.INFO)
handler = logging.FileHandler("logs.log", mode="w")
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
my_logger.addHandler(handler)


@app.route("/", methods=["get", "post"])
def index():
    my_logger.info("Page reference /")
    if request.method == "GET":
        my_logger.info("GET request")
        return (
            render_template(
                "index.html",
                img_data=None,
                not_img=False,
            ),
            200,
        )

    my_logger.info("POST request")

    start_time = time.time()

    img_file = request.files["img_file"]

    # если мы попали сюда, значит предыдущие проверки прошли успешно
    # и мы получили изображение
    # при начале оработки файла будем следить за тем,
    # что мы получили изображение, а не что-то еще
    try:
        img = Image.open(img_file.stream)
    except IOError:
        return (
            render_template(
                "index.html",
                not_img=True,
            ),
            415,
        )  # unsopported media type
        my_logger.warning("The user tried to upload a non-image")
    else:
        my_logger.info("Image uploaded successfully")

    img = convert_from_image_to_cv2(img)
    preds = model(img)
    my_logger.info("The model has processed the incoming image")

    # если алгоритм не обнаружил меню на изображении,
    # то просто возвращаем исходный объект
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
            draw_bbox(img, best_pred.bbox, color=(0, 0, 255))
    else:
        my_logger.info("The model did not find the desired object")

    img = convert_from_cv2_to_image(img)

    with BytesIO() as buf:
        img.save(buf, "jpeg")
        image_bytes = buf.getvalue()
    encoded_string = base64.b64encode(image_bytes).decode()

    duration_ms = round((time.time() - start_time) * 1000)

    my_logger.info("The server processed the received image")

    return (
        render_template(
            "index.html",
            img_data=encoded_string,
            time_inference=duration_ms,
            not_img=False,
        ),
        200,
    )


if __name__ == "__main__":
    app.run(host="localhost", debug=True)
