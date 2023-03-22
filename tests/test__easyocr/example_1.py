# https://www.youtube.com/watch?v=H_nXZSM4WiU
# https://www.youtube.com/watch?v=owiqdzha_DE
import easyocr
import numpy as np
import cv2


inp_img_path = "tests\\test_imgs\\img_1_in.jpg"
out_img_path = "tests\\test_imgs\\img_1_out.jpg"


reader = easyocr.Reader(["ru", "en"])

img = cv2.imread(out_img_path)

results = reader.readtext(img, detail=1, paragraph=False)

for (bbox, text, prob) in results:
    (tl, tr, br, bl) = bbox
    tl = (int(tl[0]), int(tl[1]))
    tr = (int(tr[0]), int(tr[1]))
    br = (int(br[0]), int(br[1]))
    bl = (int(bl[0]), int(bl[1]))

    cv2.rectangle(img, tl, br, (0, 255, 0), 2)

cv2.imshow("img", img)
cv2.waitKey(0)