import cv2
import numpy as np
from my_package import img_module, path_module


def template_matching_zncc(src_img, tmp_img):
    height, width = src_img.shape
    height_tmp, width_tmp = tmp_img.shape

    score = np.empty((height - height_tmp, width - width_tmp))

    src_img = np.array(src_img, dtype="float")
    tmp_img = np.array(tmp_img, dtype="float")

    mu_t = np.mean(tmp_img)

    for dy in range(0, height - height_tmp):
        for dx in range(0, width - width_tmp):
            roi = src_img[dy:dy + height_tmp, dx:dx + width_tmp]
            mu_r = np.mean(roi)
            roi = roi - mu_r
            tmp_img = tmp_img - mu_t

            num = np.sum(roi * tmp_img)
            den = np.sqrt(np.sum(roi ** 2)) * np.sqrt(np.sum(tmp_img ** 2))

            if den == 0:
                score[dy, dx] = 0

            score[dy, dx] = num / den

    # スコアが最大(1に最も近い)の走査位置を返す
    pt = np.unravel_index(score.argmin(), score.shape)

    return pt[1], pt[0]


input_path = path_module.input_file_path_select()[0]
img = img_module.load_img(input_path, "gray")
tmp = img_module.roi_select(img)

h, w = tmp.shape

pt = template_matching_zncc(img, tmp)

img = cv2.rectangle(img, (pt[0], pt[1]), (pt[0] + w, pt[1] + h), (0, 0, 200), 3)

cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.imwrite("C:/github/sample/python/opencv/template-matching/zncc2.png", img)