import cv2
import numpy as np
import os
import math
import img_module

# 入力画像を取得
input_img = cv2.imread(r'C:\Users\zer0\Downloads\DSC_9684.JPG')
# カーネルの大きさを指定
kernel_size = 64
# 画像のトリミング
trim_img = img_module.trim(input_img, kernel_size)
# 画像の分割と保存
img_module.split(trim_img, kernel_size, r'C:\Users\zer0\Downloads\split_image')
# ボケ量マップの作成と保存
number_of_kernel = trim_img.shape[0] // kernel_size
img_module.bokeh_detection(number_of_kernel, r'C:\Users\zer0\Downloads\split_image',
                           r'C:\Users\zer0\Downloads\output_image')
