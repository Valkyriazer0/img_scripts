"""本モジュールの説明
   ファイル操作に使用する種々の関数群
"""
import os.path
import sys
import tkinter
import glob
import time
import math
from tkinter import filedialog, messagebox


def file_path_select() -> list:
    """
    単一or複数のファイル選択ダイアログの表示

    Return
    -------
    file_path_list : list
        ファイルパス
    """
    root = tkinter.Tk()
    root.withdraw()
    file_type = [("画像ファイル", "*.jpg;*.png;*.bmp"), ("すべてのファイル", "*.*")]
    initial_dir = os.path.expanduser('~/Downloads')
    file_res = messagebox.askokcancel("入力ファイルの選択", "入力ファイルを選択してください")
    if file_res:
        file_path = filedialog.askopenfilenames(filetypes=file_type, initialdir=initial_dir)
        if file_path == "":
            sys.exit(1)
        else:
            file_path_list = list(file_path)
            root.destroy()
    else:
        sys.exit(0)
    return file_path_list


def dir_path_select(io_type: int) -> str:
    """
    ディレクトリ選択ダイアログの表示

    Parameter
    ----------
    io_type : int
        input=1, output=0

    Return
    -------
    directory_path : str
        ファイルパス
    """
    root = tkinter.Tk()
    root.withdraw()
    initial_dir = os.path.expanduser('~/Downloads')
    cvt_type_dict = {1: ("入力ファイルの保存されたディレクトリの選択", "入力ファイルの保存されたディレクトリを選択してください"),
                     0: ("出力ファイルの保存されたディレクトリの選択", "出力ファイルの保存されたディレクトリを選択してください"),
                     -1: ("処理を行うディレクトリの選択", "処理を行うディレクトリを選択してください")}
    if io_type in cvt_type_dict:
        directory_res = messagebox.askokcancel(cvt_type_dict[io_type][0], cvt_type_dict[io_type][1])
    else:
        print("入出力の設定を確認してください")
        sys.exit(1)

    if directory_res:
        directory_path = filedialog.askdirectory(initialdir=initial_dir)
        if directory_path == "":
            sys.exit(1)
        else:
            root.destroy()
    else:
        sys.exit(0)
    return directory_path


def files_rename(file_name_pattern: str = "img", delimiter: str = "_", digit: int = None, file_type: str = ".png"):
    """
    単一or複数のファイルのリネーム

    Parameter
    ----------
    file_name_pattern : str
        ファイル名
    delimiter : str
        区切り文字
    digit : int
        桁数
    file_type : str
        拡張子
    """
    dir_path = dir_path_select(-1)
    files_path = os.path.join(str(dir_path) + "/*" + file_type)
    files = glob.glob(files_path)
    print(len(files))
    if digit is None:
        digit = int(math.log10(len(files)) + 1) + 1
    else:
        pass

    file_name = file_name_pattern + delimiter + "{1:0" + str(digit) + "d}" + ".png"

    if files:
        for i, old_name in enumerate(files):
            t = os.path.getmtime(old_name)
            ts = time.strftime("%Y%m%d", time.localtime(t))
            new_name = os.path.join(str(dir_path) + "/" + file_name).format(ts, i + 1)
            os.rename(old_name, new_name)
            print(old_name + "→" + new_name)
    else:
        print("変更なし")
    return
