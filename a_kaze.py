import cv2
from my_package import img_module


def image_matching(base_img, temp_img):
    """
    画像の入力

    Parameter
    ----------
    base_img : numpy.ndarray
        元画像
    temp_img : numpy.ndarray
        テンプレート画像

    Return
    -------
    result_img : numpy.ndarray
        出力画像
    """
    feature_detector = cv2.AKAZE_create()

    kp1, des1 = feature_detector.detectAndCompute(base_img, None)
    kp2, des2 = feature_detector.detectAndCompute(temp_img, None)

    bf = cv2.BFMatcher()

    matches = bf.knnMatch(des1, des2, k=2)

    ratio = 0.5
    good_feature = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_feature.append([m])

    result_img = cv2.drawMatchesKnn(base_img, kp1, temp_img, kp2, good_feature, None, flags=2)

    return result_img


if __name__ == "__main__":
    # 画像の読み込み
    # img = cv2.imread(r"C:\Users\Valkyria\Downloads\DFD_jpg\DSC_9701_00001.jpg")
    img1 = cv2.imread(r"C:\Users\Valkyria\Downloads\DFD_jpg\DSC_9702_00001.jpg")
    img2 = img_module.roi_select(img1)

    # 特徴点マッチング
    res_img = image_matching(img1, img2)
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.imshow("result", res_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
