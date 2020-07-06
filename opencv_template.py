import cv2
import numpy as np

# 入力画像とテンプレート画像をで取得
img = cv2.imread(r'C:\Users\zer0\Downloads\matching\input.jpg')
temp = cv2.imread(r'C:\Users\zer0\Downloads\matching\template.jpg')

# グレースケール変換
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
temp = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)

# テンプレート画像の高さ・幅
h, w = temp.shape

# テンプレートマッチング（OpenCVで実装）
match = cv2.matchTemplate(gray, temp, cv2.TM_SQDIFF_NORMED)
min_value, max_value, min_pt, max_pt = cv2.minMaxLoc(match)
pt = min_pt

# テンプレートマッチングの結果を出力
cv2.rectangle(img, (pt[0], pt[1]), (pt[0] + w, pt[1] + h), (0, 0, 200), 3)
cv2.imwrite(r'C:\Users\zer0\Downloads\matching\result.jpg', img)