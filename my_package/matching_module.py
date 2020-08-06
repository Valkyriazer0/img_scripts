"""本モジュールの説明
   テンプレートマッチングに使用する種々の関数群
"""
import cv2
import numpy as np


def akaze_matching(src_img, temp_img):
    """
    A-KAZE特徴量

    Parameter
    ----------
    src_img : numpy.ndarray
        元画像
    temp_img : numpy.ndarray
        比較画像

    Return
    -------
    result_img : numpy.ndarray
        出力画像
    """
    feature_detector = cv2.AKAZE_create()

    kp1, des1 = feature_detector.detectAndCompute(src_img, None)
    kp2, des2 = feature_detector.detectAndCompute(temp_img, None)

    bf = cv2.BFMatcher()

    matches = bf.knnMatch(des1, des2, k=2)

    ratio = 0.5
    good_feature = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_feature.append([m])

    result_img = cv2.drawMatchesKnn(src_img, kp1, temp_img, kp2, good_feature, None, flags=2)
    return result_img


def zncc_matching(src_img, tmp_img):
    """
    ZNCC

    Parameters
    ----------
    src_img : numpy.ndarray
        入力画像
    tmp_img : numpy.ndarray
        テンプレート画像

    Return
    -------
    result_img : numpy.ndarray
        出力画像
    """
    height, width = src_img.shape
    height_tmp, width_tmp = tmp_img.shape

    score = np.empty((height - height_tmp, width - width_tmp))

    src_img = np.array(src_img, dtype="float")
    tmp_img = np.array(tmp_img, dtype="float")

    mu_t = np.mean(tmp_img)

    for dy in range(0, height - height_tmp):
        for dx in range(0, width - width_tmp):
            roi = src_img[dy:dy + height_tmp, dx:dx + width_tmp]
            mu_r = np.mean(roi)
            roi -= mu_r
            tmp_img -= mu_t

            num = np.sum(roi * tmp_img)
            den = np.sqrt(np.sum(roi ** 2)) * np.sqrt(np.sum(tmp_img ** 2))

            if den == 0:
                score[dy, dx] = 0

            score[dy, dx] = num / den

    pt = np.unravel_index(score.argmin(), score.shape)
    result_img = cv2.rectangle(src_img, (pt[0], pt[1]), (pt[0] + width_tmp, pt[1] + height_tmp), (0, 0, 200), 3)
    return result_img
