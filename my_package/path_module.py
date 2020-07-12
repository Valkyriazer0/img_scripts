"""本モジュールの説明
   ファイル操作に使用する種々の関数群
"""
import os.path
import sys
import tkinter
from tkinter import filedialog, messagebox


def input_file_path_select():
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
            sys.exit(0)
        else:
            file_path_list = list(file_path)
            return file_path_list
    elif not file_res:
        sys.exit(0)


def directory_path_select(io_type):
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
    if io_type == 1:
        directory_res = messagebox.askokcancel("入力ファイルの保存されたディレクトリの選択",
                                               "入力ファイルの保存されたディレクトリを選択してください")
        if directory_res:
            directory_path = filedialog.askdirectory(initialdir=initial_dir)
            if directory_path == "":
                sys.exit(0)
            else:
                return directory_path
        elif not directory_res:
            sys.exit(0)
    elif io_type == 0:
        directory_res = messagebox.askokcancel("出力ファイルを保存するディレクトリの選択",
                                               "出力ファイルを保存するディレクトリを選択してください")
        if directory_res:
            directory_path = filedialog.askdirectory(initialdir=initial_dir)
            if directory_path == "":
                sys.exit(1)
            else:
                return directory_path
        elif not directory_res:
            sys.exit(0)
    else:
        print("入出力の設定を確認してください")
        sys.exit(1)
