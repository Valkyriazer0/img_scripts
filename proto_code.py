"""本スクリプトの説明
   関数や処理のプロトタイプを作成するスクリプト
"""

import sys
from imgprocessing.decorator import logger, stop_watch, line_notify
import cv2
import numpy as np
import math
from imgprocessing import img, path
from win32api import GetSystemMetrics


def p():
    input_path = path.file_path_select()[0]
    img = img.load_img(input_path, "color_bgr")
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image = cv2.imread(r"C:\Users\zer0\Downloads\reference\DSC_8604_00001.jpg", 0)
    image1 = cv2.imread(r"C:\Users\zer0\Downloads\20190327201608.jpg")
    image2 = cv2.imread(r"C:\Users\zer0\Downloads\reference\DSC_9686.JPG")
    # print(len(image.shape))
    # print(len(image1.shape))
    # print(image.shape[0])
    # print(image.shape[1])
    img.roi_select(image1)
