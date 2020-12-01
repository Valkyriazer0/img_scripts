"""本モジュールの説明
   データ操作を行う種々の関数群
"""


def split_list(list_name, n):
    """
    リストをサブリストに分割する

    Parameter
    ----------
    list: list
        リスト
    n: int
        サブリストの要素数

    Return
    -------
    result_list : list
        分割後のリスト
    """
    for idx in range(0, len(list_name), n):
        yield list_name[idx:idx + n]
