"""本モジュールの説明
   画像処理に使用する種々の関数群
"""
import cv2
import numpy as np
import os.path
import sys
import math
import time
import tkinter
from tkinter import filedialog, messagebox
from sklearn import preprocessing
from tqdm import tqdm

# グローバル変数
drawing = False
complete_region = False
ix, iy, box_width, box_height = -1, -1, 0, 0
box = [ix, iy, box_width, box_height]


def input_file_path_select():
    """
    単一or複数のファイル選択ダイアログの表示

    Returns
    -------
    file_path_list : list
        ファイルパス
    """
    root = tkinter.Tk()
    root.withdraw()
    file_type = [("画像ファイル", "*.jpg;*.png;*.bmp"), ("すべてのファイル", "*.*")]
    initial_dir = os.path.expanduser('~/Downloads')
    file_res = messagebox.askokcancel("入力ファイルの選択", "入力ファイルを選択してください")
    if file_res:
        file_path = filedialog.askopenfilenames(filetypes=file_type, initialdir=initial_dir)
        if file_path == "":
            sys.exit(0)
        else:
            file_path_list = list(file_path)
            return file_path_list
    elif not file_res:
        sys.exit(0)


def directory_path_select(io_type):
    """
    ディレクトリ選択ダイアログの表示

    Parameters
    ----------
    io_type : int
        input=1, output=0

    Returns
    -------
    directory_path : str
        ファイルパス
    """
    root = tkinter.Tk()
    root.withdraw()
    initial_dir = os.path.expanduser('~/Downloads')
    if io_type == 1:
        directory_res = messagebox.askokcancel("入力ファイルの保存されたディレクトリの選択",
                                               "入力ファイルの保存されたディレクトリを選択してください")
        if directory_res:
            directory_path = filedialog.askdirectory(initialdir=initial_dir)
            if directory_path == "":
                sys.exit(0)
            else:
                return directory_path
        elif not directory_res:
            sys.exit(0)
    elif io_type == 0:
        directory_res = messagebox.askokcancel("出力ファイルを保存するディレクトリの選択",
                                               "出力ファイルを保存するディレクトリを選択してください")
        if directory_res:
            directory_path = filedialog.askdirectory(initialdir=initial_dir)
            if directory_path == "":
                sys.exit(1)
            else:
                return directory_path
        elif not directory_res:
            sys.exit(0)
    else:
        print("入出力の設定を確認してください")
        sys.exit(1)


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


def gamma_correction(img_name, gamma=1.0):
    """
    画像のガンマ補正

    Parameters
    ----------
    img_name : numpy.ndarray
        入力画像
    gamma : float
        ガンマ補正値

    Returns
    -------
    img_gamma : numpy.ndarray
        ガンマ補正後の画像
    """
    gamma_cvt = np.zeros((256, 1), dtype='uint8')
    for i in tqdm(range(256), desc="Gamma correct processing"):
        time.sleep(0.01)
        gamma_cvt[i][0] = 255 * (float(i) / 255) ** (1.0 / gamma)
    gamma_img = cv2.LUT(img_name, gamma_cvt)
    return gamma_img


def trim(img_name, kernel_size, output_path=None):
    """
    画像を正方形にトリミング（と保存）

    Parameters
    ----------
    img_name : numpy.ndarray
        入力画像
    kernel_size : int
        カーネルのサイズ
    output_path : str
        出力するディレクトリのパス

    Returns
    -------
    trimming_img : numpy.ndarray
        トリミング後の画像
    """
    height = (math.floor(img_name.shape[0] / kernel_size)) * kernel_size
    width = height
    height_margin = (img_name.shape[0] - height) // 2
    width_margin = (img_name.shape[1] - width) // 2
    trimming_img = img_name[height_margin:(img_name.shape[0] - height_margin),
                            width_margin:(img_name.shape[1] - width_margin)]
    if output_path is not None:
        cv2.imwrite(os.path.join(str(output_path) + "/" + "trimming.png"), trimming_img)
    return trimming_img


def split(img_name, kernel_size, output_path):
    """
    画像の分割と保存

    Parameters
    ----------
    img_name : numpy.ndarray
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
    for h_img in tqdm(np.vsplit(img_name, num_vertical_splits), desc="Image split processing"):  # 垂直方向に分割する。
        for v_img in np.hsplit(h_img, num_horizontal_splits):  # 水平方向に分割する。
            time.sleep(0.01)
            out_imgs.append(v_img)
    for i, img in enumerate(out_imgs):
        cv2.imwrite(os.path.join(str(output_path) + "/" + "split{}.png".format(i)), img)
    return


def bokeh_detection(number_of_img_divisions, input_path, output_path):
    """
    ボケ量の検出と画像の保存

    Parameters
    ----------
    number_of_img_divisions : int
        画像の分割数
    input_path : str
        カーネルのサイズ
    output_path : str
        出力するディレクトリのパス

    Returns
    -------
    bokeh_map_img : numpy.ndarray
        ボケ量マップ画像
    """
    height = width = number_of_img_divisions
    files = os.listdir(input_path)
    count = len(files)
    image_array = []
    for i in tqdm(range(count), desc="Bokeh detect processing"):
        time.sleep(0.01)
        img = cv2.imread(os.path.join(str(input_path) + "/" + "split{}.png".format(i)))
        image_array.append(cv2.Laplacian(img, cv2.CV_64F).var())
    image_array_normalize_min_max = preprocessing.minmax_scale(image_array)
    image_array_numpy = np.array(image_array_normalize_min_max)
    image_array_numpy_grayscale = image_array_numpy * 255
    bokeh_map_img = image_array_numpy_grayscale.reshape(height, width)
    cv2.imwrite(os.path.join(str(output_path) + "/" + "result.png"), bokeh_map_img)
    return bokeh_map_img


def my_mouse_callback(event, x, y):
    """
    マウスコールバック

    Parameters
    ----------
    event : int
        マウスイベントの取得
    x : int
        マウスのx座標
    y : int
        マウスのy座標
    """
    global ix, iy, box_width, box_height, box, drawing, complete_region

    if event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            box_width = x - ix
            box_height = y - iy

    elif event == cv2.EVENT_LBUTTONDOWN:  # マウス左押された時
        drawing = True

        ix = x
        iy = y
        box_width = 0
        box_height = 0

    elif event == cv2.EVENT_LBUTTONUP:  # マウス左離された時
        drawing = False
        complete_region = True

        if box_width < 0:
            ix += box_width
            box_width *= -1
        if box_height < 0:
            iy += box_height
            box_height *= -1

    box = [ix, iy, box_width, box_height]  # 切り取り範囲格納
    return


def roi_select(img_name, output_path=None):
    """
    ROIの設定（とROI画像の保存）

    Parameters
    ----------
    img_name : numpy.ndarray
        入力画像
    output_path : str
        出力するディレクトリのパス

    Returns
    -------
    roi_img : numpy.ndarray
        ROI画像
    """
    global ix, iy, box_width, box_height, box, drawing, complete_region

    roi_image = []
    source_window = "draw_rectangle"
    roi_window = "region_of_image"

    img_copy = img_name.copy()  # 画像コピー

    cv2.namedWindow(source_window)
    cv2.setMouseCallback(source_window, my_mouse_callback)

    while True:
        cv2.imshow(source_window, img_copy)

        if drawing:  # 左クリック押されてたら
            img_copy = img_name.copy()  # 画像コピー
            cv2.rectangle(img_copy, (ix, iy), (ix + box_width, iy + box_height), (0, 255, 0), 2)  # 矩形を描画

        if complete_region:  # 矩形の選択が終了したら
            complete_region = False

            roi_image = img_name[iy:iy + box_height, ix:ix + box_width]  # 元画像から選択範囲を切り取り
            cv2.imshow(roi_window, roi_image)  # 切り取り画像表示

        # キー操作
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # esc押されたら終了
            break
        elif k == ord('s') and output_path is not None:  # 's'押されたら画像を保存
            cv2.imwrite(os.path.join(str(output_path) + "/" + "roi.png"), roi_image)
            break

    cv2.destroyAllWindows()
    return roi_image
