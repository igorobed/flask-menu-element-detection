from flask import Flask, render_template, request, jsonify, make_response
from PIL import Image
from io import BytesIO
import base64
import time
import cv2
from utils import (
    my_logger,
    convert_from_image_to_cv2,
    crop_head,
    custom_crop,
    get_changed_region,
)

from utils import show_img

from detect import MyDetector, MyDetectorUI
from sklearn.cluster import AgglomerativeClustering
import io


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
def get_detection_menu():
    # img_before = request.files["img_before"]  # до нажатия бургер-меню
    # img_after = request.files["img_after"]  # после нажатия бургер-меню

    img_before = request.form.get("img_before")  # до нажатия бургер-меню
    img_after = request.form.get("img_after")

    # img_b, img_a = None, None

    img_b = base64.b64decode(img_before)
    img_a = base64.b64decode(img_after)

    # img_b = Image.open(img_before.stream)
    # img_a = Image.open(img_after.stream)

    img_b = Image.open(io.BytesIO(img_b))
    img_a = Image.open(io.BytesIO(img_a))

    img_b = convert_from_image_to_cv2(img_b)
    img_a = convert_from_image_to_cv2(img_a)

    # чтобы вернуться к оригинальным координатам
    cropped_x_y_in = {"x": 0, "y": 0}

    cropped_x_y_out = {"x": 0, "y": 0}

    img_b_crop = crop_head(img_b, cropped_x_y=cropped_x_y_in)
    img_a_crop = crop_head(img_a, cropped_x_y=cropped_x_y_out)

    menu_box, score = get_changed_region(img_b_crop, img_a_crop)

    # score выдает процент схожести двух изображений
    if menu_box is None:
        # print(f"Same image: {score}")
        return (
            jsonify({"msg": "Same image"}),
            200,
        )
    elif menu_box == {}:
        # print("Menu field not detected")
        return (
            jsonify({"msg": "Menu field not detected"}),
            200,
        )

    # вырезаем область, притерпевшую изменения
    cropped_x_y_out["x"] += menu_box["x"]
    cropped_x_y_out["y"] += menu_box["y"]
    menu_box_img = custom_crop(img_a_crop, **menu_box)

    predictions_crop = detector_ui.model(menu_box_img)

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

    max_square = 0
    max_rectangle = None
    for item in union_rectangles:
        curr_square = (item[1][0] - item[0][0]) * (item[1][1] - item[0][1])
        if curr_square > max_square:
            max_rectangle = item
            max_square = curr_square

    # после того, как обнаружили необходимую область,
    # получим список элементов, принадлежащих ей
    result_text_regions = []
    for item in predictions_crop:
        if item.class_name == "Text":
            x, y = int(item.bbox[0]), int(item.bbox[1])
            if x >= (max_rectangle[0][0] - 10) and y >= (max_rectangle[0][1] - 10):
                temp = item.bbox
                temp[0] += cropped_x_y_out["x"]
                temp[2] += cropped_x_y_out["x"]
                temp[1] += cropped_x_y_out["y"]
                temp[3] += cropped_x_y_out["y"]

                # cv2.rectangle(
                #     img_a,
                #     list(map(int, temp[:2])),
                #     list(map(int, temp[2:])),
                #     (0, 0, 255),
                #     2,
                # )

                result_text_regions.append(temp)

    # show_img(img_a)

    # нужно сделать дополнительную люоаботку найденной максимальной области
    # в частности брать не максимальную, а вторую, если она меньше не более чем на 10-15 процентов
    # и у нее в отличие от максимальной высота больше ширины а у максимальной наоборот

    # результаты нужно сортировать по оси y
    result_text_regions.sort(key=lambda x: x[1])

    return make_response(jsonify({"menu_elements": result_text_regions}), 200)


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
