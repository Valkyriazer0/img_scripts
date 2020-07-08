import cv2
import numpy as np
import img_module

gamma = 2.2

gamma_cvt = np.zeros((256, 1), dtype='uint8')

for i in range(256):
    gamma_cvt[i][0] = 255 * (float(i) / 255) ** (1.0 / gamma)

input_image_path = img_module.input_file_path_select()[0]
img_org = cv2.imread(input_image_path)

img_gamma = cv2.LUT(img_org, gamma_cvt)

cv2.namedWindow("original", cv2.WINDOW_NORMAL)
cv2.namedWindow("gamma", cv2.WINDOW_NORMAL)
cv2.imshow("original", img_org)
cv2.imshow("gamma", img_gamma)
cv2.waitKey(0)
cv2.destroyAllWindows()
