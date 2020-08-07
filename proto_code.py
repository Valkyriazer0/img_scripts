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
roi = img_module.roi_select(img)
cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.imshow("img", roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
