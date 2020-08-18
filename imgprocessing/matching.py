"""本モジュールの説明
   テンプレートマッチングに使用する種々の関数群
"""
import cv2
import numpy as np


def akaze_matching(src_img: np.ndarray, tmp_img: np.ndarray) -> np.ndarray:
    """
    A-KAZE特徴量

    Parameter
    ----------
    src_img : np.ndarray
        元画像
    temp_img : np.ndarray
        比較画像

    Return
    -------
    result_img : np.ndarray
        出力画像
    """
    akaze = cv2.AKAZE_create()

    kp1, des1 = akaze.detectAndCompute(src_img, None)
    kp2, des2 = akaze.detectAndCompute(tmp_img, None)

    bf = cv2.BFMatcher()

    matches = bf.knnMatch(des1, des2, k=2)

    ratio = 0.5
    good_feature = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_feature.append([m])

    result_img = cv2.drawMatchesKnn(src_img, kp1, tmp_img, kp2, good_feature, None, flags=2)

    img1_pt = [list(map(int, kp1[m[0].queryIdx].pt)) for m in matches]
    img2_pt = [list(map(int, kp2[m[0].trainIdx].pt)) for m in matches]

    print("src")
    print(img1_pt)
    print(len(img1_pt))
    print("tmp")
    print(img2_pt)
    print(len(img2_pt))

    return result_img


def zncc_matching(src_img: np.ndarray, tmp_img: np.ndarray) -> np.ndarray:
    """
    ZNCC

    Parameters
    ----------
    src_img : np.ndarray
        入力画像
    tmp_img : np.ndarray
        テンプレート画像

    Return
    -------
    result_img : np.ndarray
        出力画像
    """
    gray_src_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
    gray_tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_RGB2GRAY)

    h, w = gray_tmp_img.shape

    match = cv2.matchTemplate(gray_src_img, gray_tmp_img, cv2.TM_CCOEFF_NORMED)
    min_value, max_value, min_pt, max_pt = cv2.minMaxLoc(match)
    pt = max_pt

    # テンプレートマッチングの結果を出力
    result_img = cv2.rectangle(src_img, (pt[0], pt[1]), (pt[0] + w, pt[1] + h), (0, 0, 200), 3)

    return result_img


src = cv2.imread(r"C:\Users\zer0\Downloads\resize.png")
tmp = cv2.imread(r"C:\Users\zer0\Downloads\roi\roi.png")
tmp = cv2.GaussianBlur(tmp, (11, 11), 0)
result = akaze_matching(src, tmp)
cv2.imshow("img", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
