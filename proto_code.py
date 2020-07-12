"""本スクリプトの説明
   関数や処理のプロトタイプを作成するスクリプト
"""

import sys
from my_package.decorator import stop_watch
import cv2
import numpy as np
from my_package import img_module, path_module


# img = cv2.imread(r'C:\Users\zer0\Downloads\DSC_9684.JPG')
img2 = cv2.imread(r'C:\Users\zer0\Downloads\gaussian.jpg')


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
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.namedWindow("img_org", cv2.WINDOW_NORMAL)
    cv2.imshow("img", img)
    cv2.imshow("img_org", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
