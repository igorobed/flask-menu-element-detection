from flask import Flask, render_template, request, jsonify, make_response
from PIL import Image
from io import BytesIO
import base64
import time

from utils import (
    my_logger,
    convert_from_image_to_cv2,
    crop_head,
    custom_crop,
    get_changed_region,
)
from detect import MyDetector, MyDetectorUI


app = Flask(__name__)
detector = MyDetector()
detector_ui = MyDetectorUI()


@app.route("/detection", methods=["post"])
def get_detection():
    img_file = request.files["img_file"]
    try:
        img = Image.open(img_file.stream)
    except IOError:
        my_logger.warning("The user tried to upload a non-image")
        return (
            jsonify(
                {
                    "error": "Unsopported media type",
                }
            ),
            415,
        )
    else:
        my_logger.info("Image uploaded successfully")

    _, elements = detector(img, True)
    return make_response(jsonify(elements), 200)


@app.route("/detection_menu", methods=["post"])
def get_detection():
    img_before = request.files["img_before"]  # до нажатия бургер-меню
    img_after = request.files["img_after"]  # после нажатия бургер-меню

    img_b = Image.open(img_before.stream)
    img_a = Image.open(img_after.stream)

    img_b = convert_from_image_to_cv2(img_b)
    img_a = convert_from_image_to_cv2(img_a)

    cropped_x_y_in = {
        "x": 0,
        "y": 0
    }

    cropped_x_y_out = {
        "x": 0,
        "y": 0
    }

    img_b = crop_head(img_b, cropped_x_y=cropped_x_y_in)
    img_a = crop_head(img_a, cropped_x_y=cropped_x_y_out)

    menu_box, score = get_changed_region(img_b, img_a)

    # score выдает процент схожести двух изображений
    if menu_box is None:
        # print(f"Same image: {score}")
        return (
            jsonify(
                {
                    "msg": "Same image"
                }
            ),
            200,
        )
    elif menu_box == {}:
        # print("Menu field not detected")
        return (
            jsonify(
                {
                    "msg": "Same image"
                }
            ),
            200,
        )
    
    # вырезаем область, притерпевшую изменения
    menu_box_img = custom_crop(img_a, **menu_box, cropped_x_y_out)
        

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
        my_logger.warning("The user tried to upload a non-image")
        return (
            render_template(
                "index.html",
                not_img=True,
            ),
            415,
        )  # unsopported media type
    else:
        my_logger.info("Image uploaded successfully")

    # детекция искомого объекта на изображении
    # и отрисовка области с ним в случае успеха
    img = detector(img)

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
    app.run(host="0.0.0.0", port=5000)
