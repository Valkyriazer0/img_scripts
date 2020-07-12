"""本モジュールの説明
   種々のデコレータ群
"""
import time
from functools import wraps


def logger(func):
    """
    関数のロギングをするデコレータ

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
    def wrapper(*args, **kwargs):
        """
        デコレーターの中身

        Parameters
        ----------
        args : tuple
            可変長引数
        kwargs : dictionary
            可変長引数（辞書型）

        Return
        -------
        result : Any
        関数の出力
        """
        result = func(*args, **kwargs)
        print(f'{func} was called')
        return result
    return wrapper


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
    def wrapper(*args, **kwargs):
        """
        デコレーターの中身

        Parameters
        ----------
        args : tuple
            可変長引数
        kwargs : dictionary
            可変長引数（辞書型）

        Return
        -------
        result : Any
        関数の出力
        """
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start
        print(f"{func.__name__}関数は{elapsed_time}秒かかりました")
        return result
    return wrapper


def deco(func):
    """
    テンプレートデコレータ

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
    def wrapper(*args, **kwargs):
        """
        デコレーターの中身

        Parameters
        ----------
        args : tuple
            可変長引数
        kwargs : dictionary
            可変長引数（辞書型）

        Return
        -------
        result : Any
        関数の出力
        """
        print('前処理')
        func(*args, **kwargs)
        print('後処理')
        result = func(*args, **kwargs)
        return result
    return wrapper
