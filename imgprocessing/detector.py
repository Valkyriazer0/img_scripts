"""本モジュールの説明
   検出に使用する種々の関数群
"""
import os.path
import time

import cv2
import numpy as np
from sklearn import preprocessing
from tqdm import tqdm


def line_detection(img_name: np.ndarray, threshold1: int, threshold2: int) -> np.ndarray:
    """
    ハフ変換による線の検出

    Parameter
    ----------
    img_name : np.ndarray
        入力画像
    threshold1 : int
        Cannyの閾値
    threshold2 : int
        Cannyの閾値

    Return
    -------
    line_img : np.ndarray
        線検出画像
    """
    gray = cv2.cvtColor(img_name, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2, L2gradient=True)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    lines = lines.squeeze(axis=1)
    line_img = img_name

    def calc_y(x):
        """
        y座標の計算

        Parameter
        ----------
        x : int
            x座標

        Return
        -------
        y : int
            y座標
        """
        y = rho / np.sin(theta) - x * np.cos(theta) / np.sin(theta)
        return y

    for rho, theta in lines:
        h, w = img_name.shape[:2]
        if np.isclose(np.sin(theta), 0):
            x1, y1 = rho, 0
            x2, y2 = rho, h
        else:
            x1, y1 = 0, int(calc_y(0))
            x2, y2 = w, int(calc_y(w))

        line_img = cv2.line(img_name, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    return line_img


def circle_detection(img_name: np.ndarray, min_dist: int, threshold1: int, threshold2: int) -> np.ndarray:
    """
    ハフ変換による円の検出

    Parameter
    ----------
    img_name : np.ndarray
        入力画像
    min_distance : int
        円同士の最近接距離
    threshold1 : int
        Cannyの閾値
    threshold2 : int
        円の中心の検出の閾値

    Return
    -------
    circle_img : np.ndarray
        円検出画像
    """
    gray = cv2.cvtColor(img_name, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.0, minDist=min_dist, param1=threshold1, param2=threshold2)

    if circles is not None:
        circles = circles.squeeze(axis=0)
        circle_img = []
        for cx, cy, r in circles:
            circle_img = cv2.circle(img_name, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
            circle_img = cv2.circle(circle_img, (int(cx), int(cy)), 2, (0, 255, 0), 2)
    else:
        circle_img = img_name
    return circle_img


def bokeh_detection(img_divisions_count: int, input_path: str, output_path: str) -> np.ndarray:
    """
    ボケ量の検出と画像の保存

    Parameter
    ----------
    img_divisions_count : int
        画像の分割数
    input_path : str
        入力画像の保存されたディレクトリのパス
    output_path : str
        出力するディレクトリのパス

    Return
    -------
    bokeh_map_img : np.ndarray
        ボケ量マップ画像
    """
    height = width = img_divisions_count
    files = os.listdir(input_path)
    count = len(files)
    img_array = []
    for i in tqdm(range(count), desc="Bokeh detect processing"):
        time.sleep(0.01)
        img = cv2.imread(os.path.join(str(input_path) + "/" + "split{}.png".format(i)))
        img_array.append(cv2.Laplacian(img, cv2.CV_64F).var())
    img_array_normalize_min_max = preprocessing.minmax_scale(img_array)
    img_array_np = np.array(img_array_normalize_min_max)
    img_array_np_grayscale = img_array_np * 255
    bokeh_map_img = img_array_np_grayscale.reshape(height, width)
    cv2.imwrite(os.path.join(str(output_path) + "/" + "result.png"), bokeh_map_img)
    return bokeh_map_img
