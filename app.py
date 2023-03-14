from flask import Flask, render_template, request
from PIL import Image
from io import BytesIO
import base64
import time

from utils import my_logger
from detect import MyDetector


app = Flask(__name__)
detector = MyDetector()


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