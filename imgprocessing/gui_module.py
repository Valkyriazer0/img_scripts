"""本モジュールの説明
   GUI操作を行う種々の関数群
"""
import cv2
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imgprocessing.img_module import window_set
from imgprocessing.path_module import directory_path_select
from mpl_toolkits.mplot3d import Axes3D


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

    window_set(window_name, gray_img)
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


def center_of_gravity(img_name: np.ndarray, output_path: str = None) -> list:
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
        重心画像
    """
    binary_img = binary_gui(img_name, "inversion")
    contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result_img = img_name
    window_name = "center_of_gravity"
    coordinate = []

    for c in contours:
        m = cv2.moments(c)
        x = int(m["m10"] / m["m00"])
        y = int(m["m01"] / m["m00"])
        result_img = cv2.circle(img_name, (x, y), 2, (255, 0, 0), -1)
        coordinate.append({"X": x, "Y": y})

    window_set(window_name, result_img)
    cv2.imshow(window_name, result_img)

    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or cv2.getWindowProperty(window_name, 0) == -1:
            break
        elif k == ord('s'):
            if output_path is None:
                output_path = directory_path_select(0)
            else:
                pass
            keys = coordinate[0].keys()
            with open(os.path.join(str(output_path) + "/" + "coordinate.csv"), 'w', newline="") as output_file:
                dict_writer = csv.DictWriter(output_file, keys)
                dict_writer.writeheader()
                dict_writer.writerows(coordinate)
    cv2.destroyWindow(window_name)
    return coordinate
