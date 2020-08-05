import cv2

# Read image
img = cv2.imread(r"C:\Users\zer0\Downloads\DFD_jpg\DSC_9704_00001.jpg", 0)

sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

magnitude = cv2.magnitude(sobel_x, sobel_y)

magnitude_normalize = cv2.normalize(magnitude, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

cv2.namedWindow("img", cv2.WINDOW_NORMAL)

cv2.imshow("img", magnitude_normalize)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.imwrite(r"C:\Users\zer0\Downloads\DFD_jpg\2_4.png", magnitude)
