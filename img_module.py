import cv2
import numpy as np
import os
import math
from sklearn import preprocessing


# 画像を正方形にトリミング（と保存）
def trim(img_name, kernel_size, output_path=None):
    height = (math.floor(img_name.shape[0] / kernel_size)) * kernel_size
    width = height
    height_margin = (img_name.shape[0] - height) // 2
    width_margin = (img_name.shape[1] - width) // 2
    trimming_img = img_name[height_margin:(img_name.shape[0] - height_margin),
                            width_margin:(img_name.shape[1] - width_margin)]
    if output_path is not None:
        cv2.imwrite(os.path.join(str(output_path) + "/" + "trimming.png"), trimming_img)
    return trimming_img


# 画像の分割と保存
def split(img_name, kernel_size, output_path):
    vsize = hsize = kernel_size
    h, w = img_name.shape[:2]  # 画像の大きさ
    num_vsplits, num_hsplits = np.floor_divide([h, w], [vsize, hsize])  # 分割数
    # 分割する。
    out_imgs = []
    for h_img in np.vsplit(img_name, num_vsplits):  # 垂直方向に分割する。
        for v_img in np.hsplit(h_img, num_hsplits):  # 水平方向に分割する。
            out_imgs.append(v_img)
    for i, img in enumerate(out_imgs):
        cv2.imwrite(os.path.join(str(output_path) + "/" + "split{}.png".format(i)), img)
    return


# ボケ量の検出と画像の保存
def bokeh_detection(number_of_kernel, input_path, output_path):
    height = width = number_of_kernel
    files = os.listdir(input_path)
    count = len(files)
    image_array = []
    for i in range(count):
        img = cv2.imread(os.path.join(str(input_path) + "/" + "split{}.png".format(i)))
        image_array.append(cv2.Laplacian(img, cv2.CV_64F).var())
    image_array_normalize_min_max = preprocessing.minmax_scale(image_array)
    image_array_numpy = np.array(image_array_normalize_min_max)
    image_array_numpy_grayscale = image_array_numpy * 255
    image_array_reshape = image_array_numpy_grayscale.reshape(height, width)
    cv2.imwrite(os.path.join(str(output_path) + "/" + "result.png"), image_array_reshape)
    return


# グローバル変数
drawing = False
complete_region = False
ix, iy, box_width, box_height = -1, -1, 0, 0
box = [ix, iy, box_width, box_height]


# マウスコールバック
def my_mouse_callback(event, x, y, flags, param):
    global ix, iy, box_width, box_height, box, drawing, complete_region

    if event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            box_width = x - ix
            box_height = y - iy

    elif event == cv2.EVENT_LBUTTONDOWN:  # マウス左押された時
        drawing = True

        ix = x
        iy = y
        box_width = 0
        box_height = 0

    elif event == cv2.EVENT_LBUTTONUP:  # マウス左離された時
        drawing = False
        complete_region = True

        if box_width < 0:
            ix += box_width
            box_width *= -1
        if box_height < 0:
            iy += box_height
            box_height *= -1

    box = [ix, iy, box_width, box_height]  # 切り取り範囲格納
    return


# ROIの設定（とROI画像の保存）
def roi_select(img_name, output_path=None):
    global ix, iy, box_width, box_height, box, drawing, complete_region

    roi_image = []
    source_window = "draw_rectangle"
    roi_window = "region_of_image"

    img_copy = img_name.copy()  # 画像コピー

    cv2.namedWindow(source_window)
    cv2.setMouseCallback(source_window, my_mouse_callback)

    while True:
        cv2.imshow(source_window, img_copy)

        if drawing:  # 左クリック押されてたら
            img_copy = img_name.copy()  # 画像コピー
            cv2.rectangle(img_copy, (ix, iy), (ix + box_width, iy + box_height), (0, 255, 0), 2)  # 矩形を描画

        if complete_region:  # 矩形の選択が終了したら
            complete_region = False

            roi_image = img_name[iy:iy + box_height, ix:ix + box_width]  # 元画像から選択範囲を切り取り
            cv2.imshow(roi_window, roi_image)  # 切り取り画像表示

        # キー操作
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # esc押されたら終了
            break
        elif k == ord('s') and output_path is not None:  # 's'押されたら画像を保存
            cv2.imwrite(os.path.join(str(output_path) + "/" + "roi.png"), roi_image)
            break

    cv2.destroyAllWindows()
    return roi_image
