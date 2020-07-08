"""本スクリプトの説明
   関数や処理のプロトタイプを作成するスクリプト
"""
import cv2
import numpy as np

img = cv2.imread(r'C:\Users\zer0\Downloads\DSC_9684.JPG')


# def get_circle(frame, lower_color, upper_color):
#     """
#     円を検出する
#     """
#     min_radius = 25
#
#     # HSVによる画像情報に変換
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     # ガウシアンぼかしを適用して、認識精度を上げる
#     blur = cv2.GaussianBlur(hsv, (9, 9), 0)
#
#     # 指定した色範囲のみを抽出する
#     color = cv2.inRange(blur, lower_color, upper_color)
#
#     # オープニング・クロージングによるノイズ除去
#     element8 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8)
#     oc = cv2.morphologyEx(color, cv2.MORPH_OPEN, element8)
#     oc = cv2.morphologyEx(oc, cv2.MORPH_CLOSE, element8)
#
#     # 輪郭抽出
#     img, contours, hierarchy = cv2.findContours(oc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     print("{} contours.".format(len(contours)))
#
#     if len(contours) > 0:
#         # 一番大きい赤色領域を指定する
#         contours.sort(key=cv2.contourArea, reverse=True)
#         cnt = contours[0]
#
#         # 最小外接円を用いて円を検出する
#         (x, y), radius = cv2.minEnclosingCircle(cnt)
#         center = (int(x), int(y))
#         radius = int(radius)
#
#         # 円が小さすぎたら円を検出していないとみなす
#         if radius < min_radius:
#             return None
#         else:
#             return center, radius
#     else:
#         return None


def img_transform(img_name, flip=None, scale=1, rotate=0):
    """
    画像の線形変換

    Parameters
    ----------
    img_name : numpy.ndarray
        入力画像
    flip : int
        画像の反転
    flip = 0 x軸を中心に反転
    flip > 0 y軸を中心に反転
    flip < 0 点対称に反転
    scale : int
        画像の拡縮倍率
    rotate : int
        画像の回転角度
    rotate = 90 反時計回りに90度
    rotate = -90 時計回りに90度
    rotate = 180 180度回転

    Returns
    -------
    result_img : numpy.ndarray
        処理後の画像
    """
    if flip is not None:
        flip_img = cv2.flip(img_name, flip)
    else:
        flip_img = img_name

    resize_img = cv2.resize(flip_img, dsize=None, fx=scale, fy=scale)

    if rotate == 90:
        result_img = cv2.rotate(resize_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotate == -90:
        result_img = cv2.rotate(resize_img, cv2.ROTATE_90_CLOCKWISE)
    elif rotate == 180:
        result_img = cv2.rotate(resize_img, cv2.ROTATE_180)
    else:
        result_img = resize_img

    return result_img
