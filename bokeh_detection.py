"""本スクリプトの説明
   ボケ量マップの取得を行うスクリプト
"""
from imgprocessing import pre, path, io
from imgprocessing.decorator import stop_watch


@stop_watch
def main(kernel_size, gamma=1.0):
    """
    メイン関数

    Parameters
    ----------
    kernel_size : int
        カーネルサイズ
    gamma : float
        ガンマ値
    """
    # 入力画像を取得
    input_image_path = path.file_path_select()
    input_img = io.load_img(input_image_path[0], "gray")
    # 画像のガンマ補正
    img_gamma = pre.gamma_correction(input_img[0], gamma)
    # 画像のトリミング
    trim_img = pre.trim(img_gamma, kernel_size)
    # 画像の分割と保存
    output_split_img_path = path.dir_path_select(0)
    pre.split(trim_img, kernel_size, output_split_img_path)
    # ボケ量マップの作成と保存
    number_of_kernel = trim_img.shape[0] // kernel_size
    input_split_img_path = path.dir_path_select(1)
    output_bokeh_img_path = path.dir_path_select(0)
    pre.bokeh_detection(number_of_kernel, input_split_img_path,
                        output_bokeh_img_path)
    return


if __name__ == '__main__':
    main(16, 1)
