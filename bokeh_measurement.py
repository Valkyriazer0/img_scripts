import cv2
import numpy as np
import os

# 入力画像を取得
bokeh_img = cv2.imread(r'C:\Users\zer0\Downloads\DSC_9684.JPG', 0)
edge_img = cv2.imread(r'C:\Users\zer0\Downloads\images\edge.jpg')
height = bokeh_img.shape[0]
width = bokeh_img.shape[1]
print(str(bokeh_img.shape[0]) + "×" + str(bokeh_img.shape[1]))

image_array = []
for h in range(1, height-1):
    for w in range(1, width-1):
        gray_level = int(bokeh_img[h, w])
        gray_level_above = int(bokeh_img[h-1, w])
        gray_level_bottom = int(bokeh_img[h+1, w])
        gray_level_right = int(bokeh_img[h, w+1])
        gray_level_left = int(bokeh_img[h, w-1])
        ratio = (gray_level)/((gray_level_above+gray_level_bottom+gray_level_right+gray_level_left)/4)
        image_array.append(255-(abs(1-ratio)*255))

image_array_numpy = np.array(image_array)
image_array_reshape = image_array_numpy.reshape(height-2, width-2)

cv2.imwrite(r'C:\Users\zer0\Downloads\images\sample.jpg', image_array_reshape)


##外周の1画素分を除いているので取得できる画像は4015×6015になる

##4016×6016