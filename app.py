from flask import Flask, render_template, request
from PIL import Image
from io import BytesIO
import base64
import cvzone

app = Flask(__name__)
model = YOLO("static/best.pt")


@app.route("/", methods=["get", "post"])
def index():

    if request.method == "POST":
        if "img_file" not in request.files:
            pass

        img_file = request.files["img_file"]
        if img_file.filename == "":
            pass

        if img_file and check_file(img_file.filename):
            img = Image.open(img_file.stream)
            img = processed_img(img)
            with BytesIO() as buf:
                img.save(buf, 'jpeg')
                image_bytes = buf.getvalue()
            encoded_string = base64.b64encode(image_bytes).decode()

        return render_template('index.html', img_data=encoded_string), 200
    else:
        return render_template('index.html', img_data=None), 200


def check_file(filename):
    return True


def processed_img(img):



if __name__ == "__main__":
    app.run(host="localhost", debug=True)
