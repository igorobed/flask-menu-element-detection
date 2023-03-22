# https://stackoverflow.com/questions/56183201/detect-and-visualize-differences-between-two-images-with-opencv-python
from skimage.metrics import structural_similarity
import cv2
import numpy as np


# inp_img_path = "tests\\test_img_difference\\0bc3cafa-3d3e-4f07-aa21-085425e7d704.jpg"
# out_img_path = "tests\\test_img_difference\\83492498-921b-4486-ac60-5f1c19d3fbd6.jpg"
inp_img_path = "tests\\test_img_difference\\img_3_in.jpg"
out_img_path = "tests\\test_img_difference\\img_3_out.jpg"


def crop_head(img: np.ndarray) -> np.ndarray:
    """
    Обрезаем верхушку с временем, уровнем заряда,показателями телефона
    и полем с адресом страницы
    """
    return img[140:, :]


def show_img(img: np.ndarray) -> None:
    cv2.imshow("img", img)
    cv2.waitKey(0)


img_in = cv2.imread(inp_img_path)
img_out = cv2.imread(out_img_path)

img_in = crop_head(img_in)
img_out = crop_head(img_out)

img_in_gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
img_out_gray = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)

(score, diff) = structural_similarity(img_in_gray, img_out_gray, full=True)
print("Image Similarity: {:.4f}%".format(score * 100))

diff = (diff * 255).astype("uint8")
diff_box = cv2.merge([diff, diff, diff])

thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# thresh = cv2.threshold(diff, 3, 255, cv2.THRESH_BINARY)[1]

contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

mask = np.zeros(img_in.shape, dtype="uint8")
filled_after = img_out.copy()

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
    cv2.rectangle(img_out, (x, y), (x + w, y + h), (36, 255, 12), 2)

show_img(img_out)
