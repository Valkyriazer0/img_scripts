"""本モジュールの説明
   前処理に使用する種々の関数群
"""
import os.path
import sys
import time

import cv2
import numpy as np
from tqdm import tqdm
from win32api import GetSystemMetrics


def window_config(window_name: str, img_name: np.ndarray):
    """
    ウィンドウサイズの制御

    Parameter
    ----------
    window_name : str
        ウィンドウの名前
    img_name : np.ndarray
        入力画像
    """
    monitor_width = GetSystemMetrics(0)
    monitor_height = GetSystemMetrics(1)
    img_height, img_width = img_name.shape[:2]
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if img_height > img_width and img_height > monitor_height:
        window_width = img_width // -(-img_height // monitor_height)
        window_height = img_height // -(-img_height // monitor_height)
        cv2.resizeWindow(window_name, window_width, window_height)
    elif img_width > img_height and img_width > monitor_width:
        window_width = img_width // -(-img_width // monitor_width)
        window_height = img_height // -(-img_width // monitor_width)
        cv2.resizeWindow(window_name, window_width, window_height)
    else:
        cv2.resizeWindow(window_name, img_width, img_height)
    return


def color_cvt(img_name: np.ndarray, cvt_type: str = "bgr2rgb") -> np.ndarray:
    """
    色空間の変換

    Parameter
    ----------
    img_name : np.ndarray
        入力画像
    cvt_type : str
        bgr2rgb, bgr2hsv, bgr2gray, rgb2bgr,
        rgb2hsv, rgb2gray, hsv2bgr, hsv2rgb

    Return
    -------
    conversion_img : np.ndarray
        処理後の画像
    """
    cvt_type_dict = {"bgr2rgb": cv2.COLOR_BGR2RGB, "bgr2hsv": cv2.COLOR_BGR2HSV, "bgr2gray": cv2.COLOR_BGR2GRAY,
                     "rgb2bgr": cv2.COLOR_RGB2BGR, "rgb2hsv": cv2.COLOR_RGB2HSV, "rgb2gray": cv2.COLOR_RGB2GRAY,
                     "hsv2bgr": cv2.COLOR_HSV2BGR, "hsv2rgb": cv2.COLOR_HSV2RGB}
    if cvt_type in cvt_type_dict:
        cvt_img = cv2.cvtColor(img_name, cvt_type_dict[cvt_type])
    else:
        sys.exit(1)
    return cvt_img


def gray_check(img_name: np.ndarray) -> np.ndarray:
    """
    グレーイメージかどうかチェックする

    Parameter
    ----------
    img_name : np.ndarray
        入力画像

    Return
    -------
    result_img : np.ndarray
        処理後の画像
    """
    if len(img_name.shape) == 3:
        result_img = cv2.cvtColor(img_name, cv2.COLOR_BGR2GRAY)
    else:
        result_img = img_name
    return result_img


def rgb_separator(img_name: np.ndarray) -> list:
    """
    rgbを分離する

    Parameter
    ----------
    img_name : np.ndarray
        入力画像

    Return
    -------
    result_img_list : list
        rgbで分割した画像のリスト
    """
    channel_indices = range(img_name.shape[2])
    result_img_list = []
    for channel_index in channel_indices:
        channel_img = img_name[:, :, channel_index]
        result_img_list.append(channel_img)
    return result_img_list


def img_transform(img_name: np.ndarray, flip: int = None, scale: float = 1.0, rotate: int = 0) -> np.ndarray:
    """
    画像の線形変換

    Parameter
    ----------
    img_name : np.ndarray
        入力画像
    flip : int
        画像の反転
        flip = 0 x軸を中心に反転
        flip > 0 y軸を中心に反転
        flip < 0 点対称に反転
    scale : float
        画像の拡縮倍率
    rotate : int
        画像の回転角度
        rotate = 90 反時計回りに90度
        rotate = -90 時計回りに90度
        rotate = 180 180度回転

    Return
    -------
    result_img : np.ndarray
        処理後の画像
    """
    if type(flip) == int:
        flip_img = cv2.flip(img_name, flip)
    else:
        flip_img = img_name

    resize_img = cv2.resize(flip_img, dsize=None, fx=scale, fy=scale)

    rotate_type_dict = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, -90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180}
    if rotate in rotate_type_dict:
        result_img = cv2.rotate(resize_img, rotate_type_dict[rotate])
    elif rotate == 0:
        result_img = resize_img
    else:
        sys.exit(1)
    return result_img


def trim(img_name: np.ndarray, kernel_size: int, output_path: str = None) -> np.ndarray:
    """
    画像を正方形にトリミング（と保存）

    Parameter
    ----------
    img_name : np.ndarray
        入力画像
    kernel_size : int
        カーネルのサイズ
    output_path : str
        出力するディレクトリのパス

    Return
    -------
    trimming_img : np.ndarray
        トリミング後の画像
    """
    if img_name.shape[0] < img_name.shape[1]:
        height = img_name.shape[0] // kernel_size * kernel_size
        width = height
    else:
        width = img_name.shape[1] // kernel_size * kernel_size
        height = width
    height_margin = (img_name.shape[0] - height) // 2
    width_margin = (img_name.shape[1] - width) // 2
    trimming_img = img_name[height_margin:(img_name.shape[0] - height_margin),
                            width_margin:(img_name.shape[1] - width_margin)]
    if output_path is not None:
        cv2.imwrite(os.path.join(str(output_path) + "/" + "trimming.png"), trimming_img)
    return trimming_img


def split(img_name: np.ndarray, kernel_size: int, output_path: str):
    """
    画像の分割と保存

    Parameter
    ----------
    img_name : np.ndarray
        入力画像
    kernel_size : int
        カーネルのサイズ
    output_path : str
        出力するディレクトリのパス
    """
    vertical_size = horizontal_size = kernel_size
    h, w = img_name.shape[:2]  # 画像の大きさ
    num_vertical_splits, num_horizontal_splits = np.floor_divide([h, w], [vertical_size, horizontal_size])  # 分割数
    # 分割する。
    out_imgs = []
    for h_img in tqdm(np.vsplit(img_name, num_vertical_splits), desc="Image split processing"):
        for v_img in np.hsplit(h_img, num_horizontal_splits):
            time.sleep(0.01)
            out_imgs.append(v_img)
    for i, img in enumerate(tqdm(out_imgs, desc="Image save processing")):
        time.sleep(0.01)
        cv2.imwrite(os.path.join(str(output_path) + "/" + "split{}.png".format(i)), img)
    return
