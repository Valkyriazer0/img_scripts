"""本スクリプトの説明
   関数や処理のプロトタイプを作成するスクリプト
"""

import sys
from my_package.decorator import stop_watch
import cv2
import numpy as np
import math
from my_package import img_module, path_module


img = cv2.imread(r"C:\Users\Valkyria\Downloads\input.png", 0)
img2 = cv2.imread(r"C:\Users\Valkyria\Downloads\input.png")
cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.imshow("img", img)
cv2.waitKey(0)

print("a")
cv2.namedWindow("img2", cv2.WINDOW_NORMAL)
cv2.imshow("img2", img2)
cv2.waitKey(0)