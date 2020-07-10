"""本スクリプトの説明
   ボケ量マップの取得を行うスクリプト
"""
import cv2
from my_package import img_module, file_module
from my_package.decorator import stop_watch


@stop_watch
def main():
    """
    メイン関数
    """
    # 入力画像を取得
    input_image_path = file_module.input_file_path_select()[0]
    input_img = cv2.imread(input_image_path)
    # カーネルの大きさを指定
    kernel_size = 16
    # 画像のガンマ補正
    img_gamma = img_module.gamma_correction(input_img, 1 / 2.2)
    # 画像のトリミング
    trim_img = img_module.trim(img_gamma, kernel_size)
    # 画像の分割と保存
    output_split_img_path = file_module.directory_path_select(0)
    img_module.split(trim_img, kernel_size, output_split_img_path)
    # ボケ量マップの作成と保存
    number_of_kernel = trim_img.shape[0] // kernel_size
    input_split_img_path = file_module.directory_path_select(1)
    output_bokeh_img_path = file_module.directory_path_select(0)
    img_module.bokeh_detection(number_of_kernel, input_split_img_path,
                               output_bokeh_img_path)


if __name__ == '__main__':
    main()
