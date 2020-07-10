"""本モジュールの説明
   種々のデコレータ群
"""
import time
from functools import wraps


def stop_watch(func):
    """
    関数の処理時間の計測をするデコレータ

    Parameter
    ----------
    func : function
        関数

    Return
    -------
    wrapper : function
        関数
    """
    @wraps(func)
    def wrapper(*args, **kargs):
        """
        デコレーターの中身

        Parameters
        ----------
        args : tuple
            可変長引数
        kargs : dictionary
            可変長引数（辞書型）

        Return
        -------
        result : Any
        関数の出力
        """
        start = time.time()
        result = func(*args,**kargs)
        elapsed_time = time.time() - start
        print(f"{func.__name__}関数は{elapsed_time}秒かかりました")
        return result
    return wrapper
