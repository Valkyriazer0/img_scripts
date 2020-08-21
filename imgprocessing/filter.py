"""本モジュールの説明
   フィルター処理に使用する種々の関数群
"""
import cv2
import numpy as np
import sys
import time
from tqdm import tqdm
from .pre import window_config


def canny_not_binary(img_name: np.ndarray) -> np.ndarray:
    """
    2値化しないCanny法

    Parameter
    ----------
    img_name : np.ndarray
        入力画像

    Return
    -------
    canny_img : np.ndarray
        処理後の画像
    """
    sobel_x_img = cv2.Sobel(img_name, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y_img = cv2.Sobel(img_name, cv2.CV_64F, 0, 1, ksize=3)

    magnitude_img = cv2.magnitude(sobel_x_img, sobel_y_img)
    canny_img = cv2.convertScaleAbs(magnitude_img)

    window_config("canny_img", canny_img)
    cv2.imshow("canny_img", canny_img)
    cv2.waitKey(0)
    cv2.destroyWindow("canny_img")
    return canny_img


def blur_filter(img_name: np.ndarray, filter_type: str, kernel_size: int = 3) -> np.ndarray:
    """
    ぼかしフィルタ

    Parameter
    ----------
    img_name : np.ndarray
        入力画像
    filter_type : str
        average, gauss, bilateral
    kernel_size : int
        カーネルのサイズ

    Return
    -------
    result_img : np.ndarray
        処理後の画像
    """
    if filter_type == "average":
        result_img = cv2.blur(img_name, (kernel_size, kernel_size))
    elif filter_type == "gauss":
        result_img = cv2.GaussianBlur(img_name, (kernel_size, kernel_size), 0)
    elif filter_type == "bilateral":
        result_img = cv2.bilateralFilter(img_name, kernel_size, 20, 20)
    elif filter_type == "median":
        result_img = cv2.medianBlur(img_name, kernel_size)
    else:
        sys.exit(1)
    return result_img


def unsharp_masking(img_name: np.ndarray) -> np.ndarray:
    """
    鮮鋭化フィルタ

    Parameter
    ----------
    img_name : np.ndarray
        入力画像

    Return
    -------
    result_img : np.ndarray
        処理後の画像
    """
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]], np.float32)

    result_img = cv2.filter2D(img_name, -1, kernel)
    return result_img


def lowpass_filter(src, a=0.5):
    # 高速フーリエ変換(2次元)
    src = np.fft.fft2(src)

    # 画像サイズ
    h, w = src.shape

    # 画像の中心座標
    cy, cx = int(h / 2), int(w / 2)

    # フィルタのサイズ(矩形の高さと幅)
    rh, rw = int(a * cy), int(a * cx)

    # 第1象限と第3象限、第1象限と第4象限を入れ替え
    fsrc = np.fft.fftshift(src)

    # 入力画像と同じサイズで値0の配列を生成
    fdst = np.zeros(src.shape, dtype=complex)

    # 中心部分の値だけ代入（中心部分以外は0のまま）
    fdst[cy - rh:cy + rh, cx - rw:cx + rw] = fsrc[cy - rh:cy + rh, cx - rw:cx + rw]

    # 第1象限と第3象限、第1象限と第4象限を入れ替え(元に戻す)
    fdst = np.fft.fftshift(fdst)

    # 高速逆フーリエ変換
    dst = np.fft.ifft2(fdst)
    result_img = np.uint8(dst.real)

    # 実部の値のみを取り出し、符号なし整数型に変換して返す
    return result_img


def open_close_denoise(img_name: np.ndarray, denoise_type: str = "opening", kernel_size: int = 5) -> np.ndarray:
    """
    オープニング、クロージング処理によるデノイズ

    Parameter
    ----------
    img_name : np.ndarray
        入力画像（2値化画像）
    denoise_type : str
        opening, closing
    kernel_size : int
        カーネルのサイズ

    Return
    -------
    result_img : np.ndarray
        出力画像
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if denoise_type == "opening":
        result_img = cv2.morphologyEx(img_name, cv2.MORPH_OPEN, kernel)
    elif denoise_type == "closing":
        result_img = cv2.morphologyEx(img_name, cv2.MORPH_CLOSE, kernel)
    else:
        sys.exit(1)
    return result_img


def gamma_correction(img_name: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    画像のガンマ補正

    Parameter
    ----------
    img_name : np.ndarray
        入力画像
    gamma : float
        ガンマ補正値

    Return
    -------
    img_gamma : np.ndarray
        ガンマ補正後の画像
    """
    gamma_cvt = np.zeros((256, 1), dtype='uint8')
    for i in tqdm(range(256), desc="Gamma correct processing"):
        time.sleep(0.01)
        gamma_cvt[i][0] = 255 * (float(i) / 255) ** (1.0 / gamma)
    gamma_img = cv2.LUT(img_name, gamma_cvt)
    return gamma_img
