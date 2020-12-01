"""本モジュールの説明
   DLT法の計算を行う種々の関数群
"""
import re

import numpy as np

from common.path import file_path_select
from common.data import split_list


def is_float(s):
    """
    文字列を数字に変換

    Parameter
    ----------
    s : str
        文字列
    """
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True


def cal_dlt_para_from_cog(cog_coordinate: list, marker: list = (4, 5), distance: float = 4.2,
                          depth: float = 0.0) -> np.ndarray:
    """
    重心座標と実座標からDLTパラメータの計算

    Parameter
    ----------
    cog_coordinate : list
        重心座標
    marker : list
        マーカーの数
    distance : float
        マーカー同士の距離
    depth : float
        奥行き

    Return
    -------
    dlt_para : np.ndarray
        DLTパラメータ
    """
    uv_sorted = sorted(cog_coordinate, key=lambda s: s['u'])
    uv_sorted = sorted(uv_sorted, key=lambda t: t['v'])

    uv_coordinate = []
    for i in range(len(uv_sorted)):
        uv_coordinate.append(uv_sorted[i]['u'])
        uv_coordinate.append(uv_sorted[i]['v'])

    x_mat = []
    i = 0
    for row in range(marker[0]):
        for column in range(marker[1]):
            x = column * distance
            y = row * distance
            z = depth
            u = uv_coordinate[2 * i]
            v = uv_coordinate[2 * i + 1]
            pre_mat = [x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z]
            x_mat.append(pre_mat)
            pre_mat = [0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z]
            x_mat.append(pre_mat)
            i += 1
    x_mat = np.array(x_mat)

    xt = x_mat.T
    xtx = np.dot(xt, x_mat)
    i = np.eye(11)
    # 影響しない程度に小さい単位行列を足して逆行列が生成不可な場合を防ぐ
    xtx = xtx + 0.0001 * i
    xtx_inv = np.linalg.inv(xtx)
    xtx_inv_xt = np.dot(xtx_inv, xt)
    dlt_para = np.dot(xtx_inv_xt, uv_coordinate)
    return dlt_para


def cal_dlt_para_point(target: int, uv_coordinate_path: str, real_coordinate_path: str) -> np.ndarray:
    """
    画像座標と実座標からDLTパラメータの計算

    Parameter
    ----------
    target : int
        ターゲット数

    Return
    -------
    dlt_para : np.ndarray
        DLTパラメータ
    """
    real_coordinate_list = open(real_coordinate_path, 'r', encoding='utf-8')
    uv_coordinate_list = open(uv_coordinate_path, 'r', encoding='utf-8')

    delimiter = "\t|\n"
    real_coordinate = re.split(delimiter, real_coordinate_list.read())
    real_coordinate = [float(s) for s in real_coordinate if is_float(s)]
    u_mat = re.split(delimiter, uv_coordinate_list.read())
    u_mat = [float(s) for s in u_mat if is_float(s)]

    real_coordinate = list(split_list(real_coordinate, 3))

    x_mat = []
    for i in range(target):
        x = real_coordinate[i][0]
        y = real_coordinate[i][1]
        z = real_coordinate[i][2]
        u = u_mat[i]
        v = u_mat[2 * i]
        pre_mat = [x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z]
        x_mat.append(pre_mat)
        pre_mat = [0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z]
        x_mat.append(pre_mat)
    x_mat = np.array(x_mat)

    xt = x_mat.T
    xtx = np.dot(xt, x_mat)
    xtu = np.dot(xt, u_mat)
    dlt_para = np.linalg.solve(xtx, xtu)

    # i = np.eye(11)
    # # 影響しない程度に小さい単位行列を足して逆行列が生成不可な場合を防ぐ
    # xtx = xtx + 0.0001 * i
    # xtx_inv = np.linalg.inv(xtx)
    # xtx_inv_xt = np.dot(xtx_inv, xt)
    # dlt_para = np.dot(xtx_inv_xt, uv_coordinate)

    return dlt_para


def dlt_method(dlt_para1: np.ndarray, dlt_para2: np.ndarray, uv1: list, uv2: list) -> np.ndarray:
    """
    DLTパラメータと画像座標から実座標の計算

    Parameter
    ----------
    dlt_para1 : np.ndarray
        カメラ1のDLTパラメータ
    dlt_para2 : np.ndarray
        カメラ2のDLTパラメータ
    uv1 : list
        カメラ1の画像座標
    uv2 : list
        カメラ2の画像座標

    Return
    -------
    global_coordinate : np.ndarray
        実座標
    """
    l1_1, l2_1, l3_1, l4_1, l5_1, l6_1, l7_1, l8_1, l9_1, l10_1, l11_1 = dlt_para1
    l1_2, l2_2, l3_2, l4_2, l5_2, l6_2, l7_2, l8_2, l9_2, l10_2, l11_2 = dlt_para2
    u1, v1 = uv1
    u2, v2 = uv2

    u_mat = np.array([[u1 * l9_1 - l1_1, u1 * l10_1 - l2_1, u1 * l11_1 - l3_1],
                      [v1 * l9_1 - l5_1, v1 * l10_1 - l6_1, v1 * l11_1 - l7_1],
                      [u2 * l9_2 - l1_2, u2 * l10_2 - l2_2, u2 * l11_2 - l3_2],
                      [v2 * l9_2 - l5_2, v2 * l10_2 - l6_2, v2 * l11_2 - l7_2]
                      ])
    l_mat = np.array([[l4_1 - u1],
                      [l8_1 - v1],
                      [l4_2 - u2],
                      [l8_2 - v2]
                      ])

    ut = u_mat.T
    utu = np.dot(ut, u_mat)
    utl = np.dot(ut, l_mat)
    global_coordinate = np.linalg.solve(utu, utl)

    # i_mat = np.eye(3)
    # # 影響しない程度に小さい単位行列を足して逆行列が生成不可な場合を防ぐ
    # utu = utu + 0.0001 * i_mat
    # utu_inv = np.linalg.inv(utu)
    # utu_inv_ut = np.dot(utu_inv, ut)
    # global_coordinate = np.dot(utu_inv_ut, l_mat)

    return global_coordinate
