"""本モジュールの説明
   GUI操作を行う種々の関数群
"""
import csv
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from common.path import dir_path_select
from .preprocess import window_config


def roi_select(img_name: np.ndarray, output_path: str = None) -> tuple:
    """
    ROIの設定（とROI画像の保存）

    Parameter
    ----------
    img_name : np.ndarray
        入力画像
    output_path : str
        出力するディレクトリのパス

    Return
    -------
    roi_img : np.ndarray
        ROI画像
    roi : tuple
        ROI画像の全画像中での位置座標
    """

    source_window = "draw_rectangle"
    roi_window = "region_of_image"

    show_cross_hair = False
    from_center = False
    window_config(source_window, img_name)
    roi = cv2.selectROI(source_window, img_name, from_center, show_cross_hair)
    cv2.destroyWindow(source_window)
    roi_img = img_name[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
    if roi_img.size == 0:
        sys.exit(1)
    else:
        pass

    window_config(roi_window, roi_img)
    cv2.imshow(roi_window, roi_img)

    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or cv2.getWindowProperty(roi_window, 0) == -1:
            break
        elif k == ord('s'):
            if output_path is None:
                output_path = dir_path_select(0)
            else:
                pass
            cv2.imwrite(os.path.join(str(output_path) + "/" + "roi.png"), roi_img)
            break

    cv2.destroyWindow(roi_window)
    return roi_img, roi


def center_of_gravity(img_name: np.ndarray, output_path: str = None) -> tuple:
    """
    重心計算

    Parameter
    ----------
    img_name : np.ndarray
        入力画像
    output_path : str
        出力するディレクトリのパス

    Return
    -------
    coordinate : list
        重心座標
    contours_count : int
        検出した輪郭の個数
    """
    binary_img = binary_gui(img_name, "inversion")
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_count = len(contours)
    result_img = img_name
    window_name = "center_of_gravity"
    coordinate = []

    for c in contours:
        m = cv2.moments(c)
        x = int(m["m10"] / m["m00"])
        y = int(m["m01"] / m["m00"])
        result_img = cv2.circle(img_name, (x, y), 2, (255, 0, 0), -1)
        coordinate.append({"u": x, "v": y})

    window_config(window_name, result_img)
    cv2.imshow(window_name, result_img)

    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or cv2.getWindowProperty(window_name, 0) == -1:
            break
        elif k == ord('s'):
            if output_path is None:
                output_path = dir_path_select(0)
            else:
                pass

            keys = coordinate[0].keys()
            with open(os.path.join(str(output_path) + "/" + "coordinate.csv"), 'w', newline="") as output_file:
                dict_writer = csv.DictWriter(output_file, keys)
                dict_writer.writeheader()
                dict_writer.writerows(coordinate)
    cv2.destroyWindow(window_name)
    return coordinate, contours_count


def roi2cof(img_name: np.ndarray, output_path: str = None):
    """
    ROI画像を用いた重心の座標の計算

    Parameter
    ----------
    img_name : np.ndarray
        入力画像
    output_path : str
        出力するディレクトリのパス

    Return
    -------
    coordinate : list
        重心座標
    """
    roi_img, roi = roi_select(img_name)
    coordinate, contours_count = center_of_gravity(roi_img)

    for i in range(contours_count):
        coordinate[i]['u'] = coordinate[i]['v'] + roi[0]
        coordinate[i]['u'] = coordinate[i]['v'] + roi[1]

    if output_path is None:
        output_path = dir_path_select(0)
    else:
        pass

    keys = coordinate[0].keys()
    with open(os.path.join(str(output_path) + "/" + "coordinate.csv"), 'w', newline="") as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(coordinate)
    return coordinate


def three_dim_graph(x: list, y: list, z: list):
    """
    3次元グラフ

    Parameter
    ----------
    x : list
        x軸方向の要素
    y : list
        y軸方向の要素
    z : list
        z軸方向の要素
    """
    sns.set_style("darkgrid")

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.plot(x, y, z, marker="o", linestyle='None')

    plt.show()
    return


def histogram(img_name: np.ndarray, hist_type: str = "bgr"):
    """
    ヒストグラム

    Parameter
    ----------
    img_name : np.ndarray
        入力画像
    hist_type : str
        bgr, hsv, gray
    """
    if hist_type == "bgr":
        b, g, r = img_name[:, :, 0], img_name[:, :, 1], img_name[:, :, 2]
        hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
        hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
        plt.plot(hist_r, color='r', label="r")
        plt.plot(hist_g, color='g', label="g")
        plt.plot(hist_b, color='b', label="b")
        plt.legend()
        plt.show()
    elif hist_type == "hsv":
        hsv_img = cv2.cvtColor(img_name, cv2.COLOR_BGR2HSV)
        h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
        hist_h = cv2.calcHist([h], [0], None, [256], [0, 256])
        hist_s = cv2.calcHist([s], [0], None, [256], [0, 256])
        hist_v = cv2.calcHist([v], [0], None, [256], [0, 256])
        plt.plot(hist_h, color='r', label="h")
        plt.plot(hist_s, color='g', label="s")
        plt.plot(hist_v, color='b', label="v")
        plt.legend()
        plt.show()
    elif hist_type == "gray":
        gray_img = cv2.cvtColor(img_name, cv2.COLOR_BGR2GRAY)
        hist_gray = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
        plt.plot(hist_gray, color='gray', label="gray")
        plt.show()
    else:
        sys.exit(1)
    return


def binary_gui(img_name: np.ndarray, binary_type: str = None) -> np.ndarray:
    """
    GUIを用いた画像の2値化

    Parameter
    ----------
    img_name : np.ndarray
        入力画像
    binary_type : str
        inversion

    Return
    -------
    thresh_img : np.ndarray
        2値化画像
    """

    def get_threshold(val: int):
        """
        閾値を取得

        Parameter
        ----------
        val : int
            現在の閾値
        """
        nonlocal threshold
        threshold = val
        return

    trackbar_name = "trackbar"
    window_name = "thresh"
    if len(img_name.shape) == 3:
        gray_img = cv2.cvtColor(img_name, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img_name

    window_config(window_name, gray_img)
    threshold = 100
    cv2.createTrackbar(trackbar_name, window_name, threshold, 255, get_threshold)
    while True:
        ret, thresh_img = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
        cv2.imshow(window_name, thresh_img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or cv2.getWindowProperty(window_name, 0) == -1:
            break
    cv2.destroyWindow(window_name)

    if binary_type == "inversion":
        thresh_img = cv2.bitwise_not(thresh_img)
    else:
        pass
    return thresh_img
