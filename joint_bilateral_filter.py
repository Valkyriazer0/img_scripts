import numpy as np
import cv2
from my_package.decorator import stop_watch, line_notify
from my_package import img_module, path_module


# image: input image
# texture: guidance image
# sigma_s: spatial parameter (pixels)
# sigma_r: range parameter (not normalized)
def joint_bilateral_filter(image, texture, sigma_s, sigma_r):
    r = int(np.ceil(3 * sigma_s))
    # Image padding
    if image.ndim == 3:
        h, w, ch = image.shape
        i = np.pad(image, ((r, r), (r, r), (0, 0)), 'symmetric').astype(np.float32)
    elif image.ndim == 2:
        h, w = image.shape
        i = np.pad(image, ((r, r), (r, r)), 'symmetric').astype(np.float32)
    else:
        print('Input image is not valid!')
        return image
    # Check texture size and do padding
    if texture.ndim == 3:
        ht, wt, cht = texture.shape
        if ht != h or wt != w:
            print('The guidance image is not aligned with input image!')
            return image
        T = np.pad(texture, ((r, r), (r, r), (0, 0)), 'symmetric').astype(np.int32)
    elif texture.ndim == 2:
        ht, wt = texture.shape
        if ht != h or wt != w:
            print('The guidance image is not aligned with input image!')
            return image
        T = np.pad(texture, ((r, r), (r, r)), 'symmetric').astype(np.int32)
    # Pre-compute
    output = np.zeros_like(image)
    scale_factor_s = 1 / (2 * sigma_s * sigma_s)
    scale_factor_r = 1 / (2 * sigma_r * sigma_r)
    # A lookup table for range kernel
    LUT = np.exp(-np.arange(256) * np.arange(256) * scale_factor_r)
    # Generate a spatial Gaussian function
    x, y = np.meshgrid(np.arange(2 * r + 1) - r, np.arange(2 * r + 1) - r)
    kernel_s = np.exp(-(x * x + y * y) * scale_factor_s)
    # Main body
    if i.ndim == 2 and T.ndim == 2:     # I1T1 filter
        for y in range(r, r + h):
            for x in range(r, r + w):
                wgt = LUT[np.abs(T[y - r:y + r + 1, x - r:x + r + 1] - T[y, x])] * kernel_s
                output[y - r, x - r] = np.sum(wgt * i[y - r:y + r + 1, x - r:x + r + 1]) / np.sum(wgt)
    elif i.ndim == 3 and T.ndim == 2:     # I3T1 filter
        for y in range(r, r + h):
            for x in range(r, r + w):
                wgt = LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1] - T[y, x])] * kernel_s
                wacc = np.sum(wgt)
                output[y - r, x - r, 0] = np.sum(wgt * i[y - r:y + r + 1, x - r:x + r + 1, 0]) / wacc
                output[y - r, x - r, 1] = np.sum(wgt * i[y - r:y + r + 1, x - r:x + r + 1, 1]) / wacc
                output[y - r, x - r, 2] = np.sum(wgt * i[y - r:y + r + 1, x - r:x + r + 1, 2]) / wacc
    elif i.ndim == 3 and T.ndim == 3:     # I3T3 filter
        for y in range(r, r + h):
            for x in range(r, r + w):
                wgt = LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 0] - T[y, x, 0])] * \
                      LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 1] - T[y, x, 1])] * \
                      LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 2] - T[y, x, 2])] * \
                      kernel_s
                wacc = np.sum(wgt)
                output[y - r, x - r, 0] = np.sum(wgt * i[y - r:y + r + 1, x - r:x + r + 1, 0]) / wacc
                output[y - r, x - r, 1] = np.sum(wgt * i[y - r:y + r + 1, x - r:x + r + 1, 1]) / wacc
                output[y - r, x - r, 2] = np.sum(wgt * i[y - r:y + r + 1, x - r:x + r + 1, 2]) / wacc
    elif i.ndim == 2 and T.ndim == 3:     # I1T3 filter
        for y in range(r, r + h):
            for x in range(r, r + w):
                wgt = LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 0] - T[y, x, 0])] * \
                      LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 1] - T[y, x, 1])] * \
                      LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 2] - T[y, x, 2])] * \
                      kernel_s
                output[y - r, x - r] = np.sum(wgt * i[y - r:y + r + 1, x - r:x + r + 1]) / np.sum(wgt)
    else:
        print('Something wrong!')
        return image

    # return np.clip(output, 0, 255)
    return output


@line_notify
def bilateral():
    img = img_module.load_img(r"C:\Users\zer0\Downloads\output.png")
    guide_img = img_module.load_img(r"C:\Users\zer0\Downloads\resize.png", "gray")
    img_bilateral_filter = joint_bilateral_filter(img, guide_img, 1, 0.1*255)
    cv2.imwrite(r"C:\Users\zer0\Downloads\joint_bilateral_image.png", img_bilateral_filter)


if __name__ == '__main__':
    bilateral()
