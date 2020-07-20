"""本スクリプトの説明
   関数や処理のプロトタイプを作成するスクリプト
"""

import sys
from my_package.decorator import stop_watch
import cv2
import numpy as np
import math
from my_package import img_module, path_module


img = cv2.imread(r"C:\Users\zer0\Downloads\defocus_map\mitte20510004.jpg", 0)


def get_circle(frame, lower_color, upper_color):
    """
    円を検出する
    """
    min_radius = 25

    # HSVによる画像情報に変換
    hsv = img_module.load_img(r'C:\Users\zer0\Downloads\DSC_9684.JPG', "color_hsv")

    # ガウシアンぼかしを適用して、認識精度を上げる
    blur = img_module.blur_filter(hsv, "gauss", (9, 9))

    # 指定した色範囲のみを抽出する
    color = cv2.inRange(blur, lower_color, upper_color)

    # オープニング・クロージングによるノイズ除去
    element8 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8)
    oc = cv2.morphologyEx(color, cv2.MORPH_OPEN, element8)
    oc = cv2.morphologyEx(oc, cv2.MORPH_CLOSE, element8)

    # 輪郭抽出
    img, contours, hierarchy = cv2.findContours(oc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("{} contours.".format(len(contours)))

    if len(contours) > 0:
        # 一番大きい赤色領域を指定する
        contours.sort(key=cv2.contourArea, reverse=True)
        cnt = contours[0]

        # 最小外接円を用いて円を検出する
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)

        # 円が小さすぎたら円を検出していないとみなす
        if radius < min_radius:
            return None
        else:
            return center, radius
    else:
        return None


@stop_watch
def main():
    gauss_img = img_module.blur_filter(img, "gauss", 25)
    canny_img = cv2.Canny(img, threshold1=30, threshold2=60)
    canny_gauss_img = cv2.Canny(gauss_img, threshold1=30, threshold2=60)
    output_img = canny_img / canny_gauss_img
    # output_img = cv2.Laplacian(output_img, cv2.CV_64F)

    cv2.namedWindow("output_img", cv2.WINDOW_NORMAL)
    cv2.imshow("output_img", output_img)
    # cv2.namedWindow("img_org", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("img_filter", cv2.WINDOW_NORMAL)
    # cv2.imshow("img_org", canny_img)
    # cv2.imshow("img_filter", canny_gauss_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


gauss_img = img_module.blur_filter(img, "gauss", 25)
img_difference = cv2.subtract(gauss_img, img)
img_difference_normalize = cv2.normalize(img_difference, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.imshow("img", img_difference_normalize)
cv2.waitKey(0)
cv2.destroyAllWindows()
