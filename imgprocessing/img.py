"""本モジュールの説明
   画像処理に使用する種々の関数群
"""
import cv2
import numpy as np
import os.path
import sys
import time
from sklearn import preprocessing
from tqdm import tqdm
from win32api import GetSystemMetrics
from imgprocessing.path import file_path_select, dir_path_select


def load_img(input_img_path: str = None, img_type: str = "color_bgr") -> list:
    """
    画像の入力

    Parameter
    ----------
    input_img_path : str
        入力する画像のパス
    img_type : str
        color_bgr, color_rgb, color_hsv, gray

    Return
    -------
    input_img_list : list
        入力画像のリスト
    """
    if input_img_path is None:
        input_img_path_list = file_path_select()
    else:
        input_img_path_list = input_img_path

    input_img_list = []
    for img_path in input_img_path_list:
        input_img = cv2.imread(img_path)
        cvt_type_dict = {"color_rgb": cv2.COLOR_BGR2RGB, "color_hsv": cv2.COLOR_BGR2HSV, "gray": cv2.COLOR_BGR2GRAY}
        if img_type in cvt_type_dict:
            input_img = cv2.cvtColor(input_img, cvt_type_dict[img_type])
        elif img_type == "color_bgr":
            pass
        else:
            sys.exit(1)
        input_img_list.append(input_img)
    return input_img_list


def window_set(window_name: str, img_name: np.ndarray):
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
    window_set(source_window, img_name)
    roi = cv2.selectROI(source_window, img_name, from_center, show_cross_hair)
    cv2.destroyWindow(source_window)
    roi_img = img_name[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
    if roi_img.size == 0:
        sys.exit(1)
    else:
        pass

    window_set(roi_window, roi_img)
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


def canny_not_binary(img_name: np.ndarray) -> np.ndarray:
    """
    2値化しないCanny法

    Parameter
    ----------
    img_name : np.ndarray
        入力画像

    Return
    -------
    canny_img : np.ndarray
        処理後の画像
    """
    sobel_x_img = cv2.Sobel(img_name, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y_img = cv2.Sobel(img_name, cv2.CV_64F, 0, 1, ksize=3)

    magnitude_img = cv2.magnitude(sobel_x_img, sobel_y_img)
    canny_img = cv2.convertScaleAbs(magnitude_img)

    window_set("canny_img", canny_img)
    cv2.imshow("canny_img", canny_img)
    cv2.waitKey(0)
    cv2.destroyWindow("canny_img")
    return canny_img


def blur_filter(img_name: np.ndarray, filter_type: str, kernel_size: int = 3) -> np.ndarray:
    """
    ぼかしフィルタ

    Parameter
    ----------
    img_name : np.ndarray
        入力画像
    filter_type : str
        average, gauss, bilateral
    kernel_size : int
        カーネルのサイズ

    Return
    -------
    result_img : np.ndarray
        処理後の画像
    """
    if filter_type == "average":
        result_img = cv2.blur(img_name, (kernel_size, kernel_size))
    elif filter_type == "gauss":
        result_img = cv2.GaussianBlur(img_name, (kernel_size, kernel_size), 0)
    elif filter_type == "bilateral":
        result_img = cv2.bilateralFilter(img_name, kernel_size, 20, 20)
    elif filter_type == "median":
        result_img = cv2.medianBlur(img_name, kernel_size)
    else:
        sys.exit(1)
    return result_img


def highpass_filter(img_name: np.ndarray) -> np.ndarray:
    """
    ハイパスフィルタ

    Parameter
    ----------
    img_name : np.ndarray
        入力画像

    Return
    -------
    result_img : np.ndarray
        処理後の画像
    """
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]], np.float32)

    result_img = cv2.filter2D(img_name, -1, kernel)
    return result_img


def open_close_denoise(img_name: np.ndarray, denoise_type: str = "opening", kernel_size: int = 5) -> np.ndarray:
    """
    オープニング、クロージング処理によるデノイズ

    Parameter
    ----------
    img_name : np.ndarray
        入力画像（2値化画像）
    denoise_type : str
        opening, closing
    kernel_size : int
        カーネルのサイズ

    Return
    -------
    result_img : np.ndarray
        出力画像
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if denoise_type == "opening":
        result_img = cv2.morphologyEx(img_name, cv2.MORPH_OPEN, kernel)
    elif denoise_type == "closing":
        result_img = cv2.morphologyEx(img_name, cv2.MORPH_CLOSE, kernel)
    else:
        sys.exit(1)
    return result_img


def gamma_correction(img_name: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    画像のガンマ補正

    Parameter
    ----------
    img_name : np.ndarray
        入力画像
    gamma : float
        ガンマ補正値

    Return
    -------
    img_gamma : np.ndarray
        ガンマ補正後の画像
    """
    gamma_cvt = np.zeros((256, 1), dtype='uint8')
    for i in tqdm(range(256), desc="Gamma correct processing"):
        time.sleep(0.01)
        gamma_cvt[i][0] = 255 * (float(i) / 255) ** (1.0 / gamma)
    gamma_img = cv2.LUT(img_name, gamma_cvt)
    return gamma_img


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
