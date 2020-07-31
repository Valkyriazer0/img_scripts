import cv2
import numpy as np
import matplotlib.pyplot as plt
from my_package import img_module

# Read image
img = cv2.imread(r"C:\Users\zer0\Downloads\DFD_jpg\DSC_9704_00001.jpg", 0)
# img = img_module.img_transform(img, scale=0.5)
# img = img_module.blur_filter(img, "gauss", 5)
# gauss_img = img_module.blur_filter(img, "gauss", 5)

# Sobel Operation
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
# gauss_sobel_x = cv2.Sobel(gauss_img, cv2.CV_64F, 1, 0, ksize=3)
# gauss_sobel_y = cv2.Sobel(gauss_img, cv2.CV_64F, 0, 1, ksize=3)

# Caluclate Magnitude
magnitude = cv2.magnitude(sobel_x, sobel_y)
# gauss_magnitude = np.sqrt(gauss_sobel_x ** 2 + gauss_sobel_y ** 2)
magnitude_normalize = cv2.normalize(magnitude, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
# gauss_magnitude_normalize = cv2.normalize(gauss_magnitude, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

# edge_difference = cv2.subtract(gauss_magnitude_normalize, magnitude_normalize)
# edge_difference = np.clip(edge_difference * 255, 0, 255).astype(np.uint8)
# edge_difference = cv2.bitwise_not(edge_difference)
# edge_difference = img_module.blur_filter(edge_difference, "average")

# edge_ratio = division1.img_ratio(magnitude, gauss_magnitude)
# edge_ratio = np.clip(edge_ratio * 255, 0, 255).astype(np.uint8)
# edge_ratio_normalize = cv2.normalize(edge_ratio, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

cv2.namedWindow("img", cv2.WINDOW_NORMAL)
# cv2.namedWindow("gauss_img", cv2.WINDOW_NORMAL)
# cv2.namedWindow("edge_difference", cv2.WINDOW_NORMAL)
# cv2.namedWindow("edge_ratio", cv2.WINDOW_NORMAL)

cv2.imshow("img", magnitude_normalize)
# cv2.imshow("gauss_img", gauss_magnitude_normalize)
# cv2.imshow("edge_difference", edge_difference)
# cv2.imshow("edge_ratio", edge_ratio)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(r"C:\Users\zer0\Downloads\DFD_jpg\2_4.png", magnitude)
# cv2.imwrite(r"C:\Users\zer0\Downloads\gauss_canny_not_binarization.png", gauss_magnitude)
# cv2.imwrite(r"C:\Users\zer0\Downloads\edge_difference.png", edge_difference)
# cv2.imwrite(r"C:\Users\zer0\Downloads\edge_ratio.png", edge_ratio)
