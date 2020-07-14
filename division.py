import numpy as np
import cv2
from my_package import img_module, path_module


def img_ratio(numerator, denominator):
    non_zero = denominator != 0
    result_img = np.zeros(numerator.shape)
    result_img[non_zero] = numerator[non_zero] / denominator[non_zero]
    result_img[~non_zero] = 1
    return result_img


"""
入力画像→ガウシアンカーネル→canny（ここまでは精度は微妙だが可能）
入力画像　　　　　　　　　 →canny
二枚のエッジ画像の比を取得→疎なボケ量マップ（これができてるのか不明）
→Joint Bilateral Filter（ノイズ除去）→matting Laplacian（密なボケ量マップ）
"""


input_image_path = path_module.input_file_path_select()[0]
img = img_module.load_img(input_image_path)
# img = img_module.img_transform(img, scale=0.1)

threshold1 = 30
threshold2 = 30

# img = img_module.blur_filter(img, "gauss", 5)
gauss_img = img_module.blur_filter(img, "gauss", 5)
canny_img = cv2.Canny(img, threshold1, threshold2)
canny_gauss_img = cv2.Canny(gauss_img, threshold1, threshold2)

cv2.namedWindow("canny_img", cv2.WINDOW_NORMAL)
cv2.imshow("canny_img", canny_img)
cv2.namedWindow("canny_gauss_img", cv2.WINDOW_NORMAL)
cv2.imshow("canny_gauss_img", canny_gauss_img)

img = img_ratio(canny_img, canny_gauss_img)
cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
