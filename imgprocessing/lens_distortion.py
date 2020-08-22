"""本モジュールの説明
   テンプレートマッチングに使用する種々の関数群
"""
import os

import cv2
import numpy as np

from imgprocessing.io import load_img


def lens_distortion(square_size: float = 2.1, pattern_size: list = (9, 6), reference_img: int = 38):
    """
    レンズ歪パラメータの計算

    Parameter
    ----------
    square_size : float
        チェスボードのサイズ
    pattern_size : list
        格子点の数
    reference_img : int
        リファレンス画像の枚数
    """
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size
    obj_points = []
    img_points = []

    imgs = load_img()

    i = 0
    while len(obj_points) < reference_img:
        img = imgs[i]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        i += 1

        ret, corner = cv2.findChessboardCorners(gray, pattern_size)
        if ret:
            print("detected corner!")
            print(str(len(obj_points) + 1) + "/" + str(reference_img))
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            cv2.cornerSubPix(gray, corner, (5, 5), (-1, -1), term)
            img_points.append(corner.reshape(-1, 2))
            obj_points.append(pattern_points)

    print("calculating camera parameter...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, imgs[0].shape[::-1], None, None)

    new_path = "../numpy_file"
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    else:
        pass

    np.save(r"../numpy_file/mtx", mtx)
    np.save(r"../numpy_file/dist", dist.ravel())
    print("RMS = ", ret)
    print("mtx = \n", mtx)
    print("dist = ", dist.ravel())
    return


def calibrate_img(img_name: np.ndarray, output_path: str = None) -> np.ndarray:
    """
    レンズ歪パラメータによるレンズ歪補正

    Parameter
    ----------
    img_name : np.ndarray
        入力画像
    output_path : str
        出力するディレクトリのパス

    Return
    -------
    result_img : np.ndarray
        出力画像
    """
    result_img = cv2.undistort(img_name, r"../numpy_file/mtx.npy", r"../numpy_file/dist.npy")
    if output_path is not None:
        cv2.imwrite(os.path.join(str(output_path) + "/" + "correction.png"), result_img)
    return result_img
