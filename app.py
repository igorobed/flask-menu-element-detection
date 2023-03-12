from flask import Flask, render_template, request
from PIL import Image
from io import BytesIO
import base64
from utils import (
    check_file,
    convert_from_cv2_to_image,
    convert_from_image_to_cv2,
)
from detect import model

app = Flask(__name__)


@app.route("/", methods=["get", "post"])
def index():

    if request.method == "POST":
        if "img_file" not in request.files:
            pass

        img_file = request.files["img_file"]
        if img_file.filename == "":
            pass

        if not (img_file and check_file(img_file.filename)):
            # тут надо вернуть инфу о некорректных данных от пользователя
            # убрать следующий if
            pass

        if img_file and check_file(img_file.filename):
            img = Image.open(img_file.stream)

            img = convert_from_image_to_cv2(img)
            preds = model(img)

            # часть ниже должна запускаться, если в preds больше одного элемента
            # нужна обработка ошибки, что у нас ничего не нашлось.....такое вообще может быть?

            if len(preds) > 0:
                max_conf = max([item.confidence for item in preds])
            
                # может быть только один элемент меню
                preds_for_remove = [item for item in preds if item.confidence != max_conf]

                # удаляем предсказания, в которых мы менее уверены
                for item_remove in preds_for_remove:
                    preds.remove(item_remove)

                preds.draw(img)

            img = convert_from_cv2_to_image(img)

            with BytesIO() as buf:
                img.save(buf, 'jpeg')
                image_bytes = buf.getvalue()
            encoded_string = base64.b64encode(image_bytes).decode()

        return render_template('index.html', img_data=encoded_string), 200
    else:
        return render_template('index.html', img_data=None), 200


if __name__ == "__main__":
    app.run(host="localhost", debug=True)
