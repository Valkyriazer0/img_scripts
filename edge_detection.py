import cv2
import numpy as np
import os

# 入力画像を取得
img = cv2.imread(r'C:\Users\zer0\Downloads\DSC_9684.JPG', 0)

threshold1 = 30
threshold2 = 30
edge_img = cv2.Canny(img, threshold1, threshold2)

path = r'C:\Users\zer0\Downloads\images'
cv2.imwrite(os.path.join(path , 'edge.jpg'), edge_img)
cv2.waitKey(0)