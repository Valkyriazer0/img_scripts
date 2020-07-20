import cv2

img0 = cv2.imread(r"C:\Users\zer0\Downloads\DFD_jpg\0.png", 0)
img1 = cv2.imread(r"C:\Users\zer0\Downloads\DFD_jpg\1.png", 0)
img2 = cv2.imread(r"C:\Users\zer0\Downloads\DFD_jpg\2.png", 0)
img3 = cv2.imread(r"C:\Users\zer0\Downloads\DFD_jpg\3.png", 0)
img4 = cv2.imread(r"C:\Users\zer0\Downloads\DFD_jpg\4.png", 0)
img5 = cv2.imread(r"C:\Users\zer0\Downloads\DFD_jpg\5.png", 0)
img6 = cv2.imread(r"C:\Users\zer0\Downloads\DFD_jpg\6.png", 0)
img7 = cv2.imread(r"C:\Users\zer0\Downloads\DFD_jpg\7.png", 0)
img8 = cv2.imread(r"C:\Users\zer0\Downloads\DFD_jpg\8.png", 0)
img9 = cv2.imread(r"C:\Users\zer0\Downloads\DFD_jpg\9.png", 0)
img_temp = cv2.imread(r"C:\Users\zer0\Downloads\DFD_jpg\temp.png", 0)
img = cv2.imread(r"C:\Users\zer0\Downloads\DFD_jpg\temp.png")

# input_path = path_module.directory_path_select(0)
# files = os.listdir(input_path)
# count = len(files)
# for i in tqdm(range(count), desc="Image load processing"):
#     img = cv2.imread(os.path.join(str(input_path) + "/" + "{}.png".format(i)))

for x in range(img0.shape[0]):
    for y in range(img0.shape[1]):
        if img1[x, y] > img0[x, y]:
            img[x, y] = (230, 0, 25)
            img_temp[x, y] = img1[x, y]
        else:
            img[x, y] = (255, 0, 0)
            img_temp[x, y] = img0[x, y]

for x in range(img0.shape[0]):
    for y in range(img0.shape[1]):
        if img2[x, y] > img_temp[x, y]:
            img[x, y] = (205, 0, 50)
            img_temp[x, y] = img2[x, y]
        else:
            pass

for x in range(img0.shape[0]):
    for y in range(img0.shape[1]):
        if img3[x, y] > img_temp[x, y]:
            img[x, y] = (180, 0, 75)
            img_temp[x, y] = img3[x, y]
        else:
            pass

for x in range(img0.shape[0]):
    for y in range(img0.shape[1]):
        if img4[x, y] > img_temp[x, y]:
            img[x, y] = (155, 0, 100)
            img_temp[x, y] = img4[x, y]
        else:
            pass

for x in range(img0.shape[0]):
    for y in range(img0.shape[1]):
        if img5[x, y] > img_temp[x, y]:
            img[x, y] = (130, 0, 125)
            img_temp[x, y] = img5[x, y]
        else:
            pass

for x in range(img0.shape[0]):
    for y in range(img0.shape[1]):
        if img6[x, y] > img_temp[x, y]:
            img[x, y] = (105, 0, 150)
            img_temp[x, y] = img6[x, y]
        else:
            pass

for x in range(img0.shape[0]):
    for y in range(img0.shape[1]):
        if img7[x, y] > img_temp[x, y]:
            img[x, y] = (80, 0, 175)
            img_temp[x, y] = img7[x, y]
        else:
            pass

for x in range(img0.shape[0]):
    for y in range(img0.shape[1]):
        if img8[x, y] > img_temp[x, y]:
            img[x, y] = (55, 0, 200)
            img_temp[x, y] = img8[x, y]
        else:
            pass

for x in range(img0.shape[0]):
    for y in range(img0.shape[1]):
        if img9[x, y] > img_temp[x, y]:
            img[x, y] = (30, 0, 225)
        else:
            pass

cv2.imwrite(r"C:\Users\zer0\Downloads\DFD_jpg\output.png", img)
