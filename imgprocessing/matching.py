"""本モジュールの説明
   テンプレートマッチングに使用する種々の関数群
"""
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .preprocess import gray_check


def feature_matching(src_img: np.ndarray, tmp_img: np.ndarray, feature_type: str = "akaze") -> np.ndarray:
    """
    A-KAZE特徴量

    Parameter
    ----------
    src_img : np.ndarray
        元画像
    tmp_img : np.ndarray
        比較画像
    feature_type : str
        特徴量の種類

    Return
    -------
    result_img : np.ndarray
        出力画像
    """
    if feature_type == "akaze":
        detector = cv2.AKAZE_create()
    elif feature_type == "orb":
        detector = cv2.ORB_create()
    else:
        sys.exit(1)

    kp1, des1 = detector.detectAndCompute(src_img, None)
    kp2, des2 = detector.detectAndCompute(tmp_img, None)

    bf = cv2.BFMatcher()

    matches = bf.knnMatch(des1, des2, k=2)

    ratio = 0.5
    good_feature = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_feature.append([m])

    result_img = cv2.drawMatchesKnn(src_img, kp1, tmp_img, kp2, good_feature, None, flags=2)

    src_img_pt = [list(map(int, kp1[m[0].queryIdx].pt)) for m in matches]
    tmp_img_pt = [list(map(int, kp2[m[0].trainIdx].pt)) for m in matches]

    print(feature_type)
    print("src" + str(src_img_pt))
    print(len(src_img_pt))
    print("tmp" + str(tmp_img_pt))
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
    src_img = gray_check(src_img)
    tmp_img = gray_check(tmp_img)

    h, w = tmp_img.shape

    match = cv2.matchTemplate(src_img, tmp_img, cv2.TM_CCOEFF_NORMED)
    min_value, max_value, min_pt, max_pt = cv2.minMaxLoc(match)
    pt = max_pt

    result_img = cv2.rectangle(src_img, (pt[0], pt[1]), (pt[0] + w, pt[1] + h), (0, 0, 200), 3)

    return result_img


def poc(src_img: np.ndarray, tmp_img: np.ndarray) -> tuple:
    """
    位相限定相関法

    Parameter
    ----------
    src_img : np.ndarray
        元画像
    temp_img : np.ndarray
        比較画像

    Return
    -------
    shift : tuple
        シフト量
    correlation : float
        相関係数
    """
    gray_src_img = gray_check(src_img)
    gray_tmp_img = gray_check(tmp_img)

    h, w = gray_src_img.shape
    hy = np.hanning(h)
    hx = np.hanning(w)
    hw = hy.reshape(h, 1) * hx

    f = np.fft.fft2(gray_src_img * hw)
    g = np.fft.fft2(gray_tmp_img * hw)
    g_ = np.conj(g)
    r = f * g_ / np.abs(f * g_)
    fft_r = np.real(np.fft.ifft2(r))

    x, y = np.mgrid[:fft_r.shape[0], :fft_r.shape[1]]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    surf = ax.plot_surface(x, y, fft_r, cmap='bwr', linewidth=0)
    fig.colorbar(surf)
    fig.show()
    plt.show()

    peak = np.unravel_index(np.argmax(fft_r), fft_r.shape)
    shift = [peak[1], peak[0]]
    shift = tuple(shift)
    correlation = fft_r[peak]
    return shift, correlation


def ripoc(src_img: np.ndarray, tmp_img: np.ndarray, r: int = None) -> tuple:
    """
    回転不変位相限定相関法

    Parameter
    ----------
    src_img : np.ndarray
        元画像
    temp_img : np.ndarray
        比較画像
    r : int
        半径

    Return
    -------
    shift : tuple
        シフト量
    angle : float
        回転量
    scale : float
        拡大率
    correlation : float
        相関係数
    """
    gray_src_img = gray_check(src_img)
    gray_tmp_img = gray_check(tmp_img)

    f = np.asarray(gray_src_img, 'float')
    g = np.asarray(gray_tmp_img, 'float')

    h, w = f.shape
    hy = np.hanning(h)
    hx = np.hanning(w)
    hw = hy.reshape(h, 1) * hx

    f_a = np.fft.fftshift(np.log(np.abs(np.fft.fft2(f * hw))))
    f_b = np.fft.fftshift(np.log(np.abs(np.fft.fft2(g * hw))))

    if not r:
        l2 = np.sqrt(w * w + h * h)
        r = l2 / np.log(l2)

    center = (w / 2, h / 2)
    flags = cv2.INTER_LANCZOS4 + cv2.WARP_POLAR_LOG
    p_a = cv2.warpPolar(f_a, (w, h), center, r, flags)
    p_b = cv2.warpPolar(f_b, (w, h), center, r, flags)
    (x, y), e = cv2.phaseCorrelate(p_a, p_b, hw)

    angle = y * 360 / h
    scale = np.e ** (x / r)
    m = cv2.getRotationMatrix2D(center, angle, scale)
    t_b = cv2.warpAffine(g, m, (w, h))
    shift, correlation = cv2.phaseCorrelate(f, t_b, hw)
    return shift, angle, scale, correlation
