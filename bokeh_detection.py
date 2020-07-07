"""本スクリプトの説明
   ボケ量マップの取得
"""
import cv2
import img_module


# 入力画像を取得
input_image_path = img_module.file_path_select("入力画像の選択", "入力画像を選択してください")[0]
input_img = cv2.imread(input_image_path)
# カーネルの大きさを指定
kernel_size = 64
# 画像のトリミング
trim_img = img_module.trim(input_img, kernel_size)
# 画像の分割と保存
output_split_image_path = img_module.directory_path_select("出力画像を保存するディレクトリの選択",
                                                           "出力画像を保存するディレクトリを選択してください")
img_module.split(trim_img, kernel_size, output_split_image_path)
# ボケ量マップの作成と保存
number_of_kernel = trim_img.shape[0] // kernel_size
input_split_image_path = img_module.directory_path_select("入力画像の保存されたディレクトリの選択",
                                                          "入力画像の保存されたディレクトリを選択してください")
output_bokeh_image_path = img_module.file_path_select("出力画像を保存するディレクトリの選択",
                                                      "出力画像を保存するディレクトリを選択してください")
img_module.bokeh_detection(number_of_kernel, input_split_image_path,
                           output_bokeh_image_path)
