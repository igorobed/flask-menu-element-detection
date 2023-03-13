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
import time
from cvu.utils.draw import draw_bbox

app = Flask(__name__)


@app.route("/", methods=["get", "post"])
def index():
    if request.method == "GET":
        return render_template(
            'index.html',
            img_data=None
            ), 200
    
    start_time = time.time()

    img_file = request.files["img_file"]

    if not check_file(img_file.filename):
        # надо вернуть информацию о некорректном формате данных и соответствующий код
        pass

    # если мы попали сюда, значит предыдущие проверки прошли успешно и мы получили изображение
    img = Image.open(img_file.stream)
    img = convert_from_image_to_cv2(img)
    preds = model(img)

    # если алгоритм не обнаружил меню на изображении, то просто возвращаем исходный объект
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
    
    img = convert_from_cv2_to_image(img)

    with BytesIO() as buf:
        img.save(buf, 'jpeg')
        image_bytes = buf.getvalue()
    encoded_string = base64.b64encode(image_bytes).decode()

    duration_ms = round((time.time() - start_time) * 1000)

    return render_template(
        'index.html',
        img_data=encoded_string,
        time_inference=duration_ms,
        ), 200


if __name__ == "__main__":
    app.run(host="localhost", debug=True)
