"""本モジュールの説明
   ファイルの出入力に使用する種々の関数群
"""
import math
import os.path
import sys

import cv2

from common.path import dir_path_select, file_path_select


def load_img(input_img_path: str = None, img_type: str = "color_bgr") -> list:
    """
    単一or複数の画像の入力

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
    img_path_list = []
    if input_img_path is None:
        img_path_list = file_path_select()
    else:
        img_path_list.append(input_img_path)

    input_img_list = []
    for img_path in img_path_list:
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


def load_video(input_video_path: str = None) -> list:
    """
    動画を連番画像として入力

    Parameter
    ----------
    input_video_path : str
        入力する動画のパス

    Return
    -------
    input_img_list : list
        連番画像のリスト
    """
    if input_video_path is None:
        video_path = file_path_select()[0]
    else:
        video_path = input_video_path

    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        pass
    else:
        sys.exit(1)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_list = []
    for i in range(frame_count):
        ret, frame = cap.read()
        frame_list.append(frame)

    return frame_list


def save_sequence_img(img_list: list, base_name: str = "", delimiter: str = "_", file_type: str = ".png",
                      output_path: str = None):
    """
    複数画像を連番で保存

    Parameter
    ----------
    input_list : list
        入力する連番画像のリスト
    base_name : str
        ファイル名
    delimiter : str
        区切り文字
    file_type : str
        拡張子
    output_path : str
        出力するディレクトリのパス
    """
    if output_path is None:
        output_path = dir_path_select(0)
    else:
        pass

    digit = int(math.log10(len(img_list)) + 1) + 1
    file_name = base_name + delimiter + "{0:0" + str(digit) + "d}" + file_type

    for i, img in enumerate(img_list):
        cv2.imwrite(os.path.join(str(output_path) + "/" + file_name).format(i + 1), img)
    return
