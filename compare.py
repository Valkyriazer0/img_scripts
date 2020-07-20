import cv2
import numpy as np
from my_package import img_module

img = cv2.imread(r"C:\Users\zer0\Downloads\DFD_jpg\1024.png", 0)

roi_img = img_module.roi_select(img)
pixel_value_max = np.max(roi_img)
print(pixel_value_max)
