import cv2

img0 = cv2.imread(r"C:\Users\zer0\Downloads\DFD_jpg\DSC_9695_00001.jpg")
img9 = cv2.imread(r"C:\Users\zer0\Downloads\DFD_jpg\DSC_9704_00001.jpg")

img = cv2.addWeighted(img0, 0.5, img9, 0.5, 0)

cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(r"C:\Users\zer0\Downloads\DFD_jpg\blend.png", img)
