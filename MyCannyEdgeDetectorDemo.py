import math
from collections import deque

import numpy as np
from numba import jit
from skimage import io
from skimage.color import rgb2gray


def get_gaussian_kernel(sigma, size=0):
    if size == 0:
        size = int(1 + 2 * math.ceil(sigma * np.sqrt(np.log(400))))
    mid = size // 2
    kernel = np.zeros((size, 1))
    dkernel = np.zeros((size, 1))
    for i in range(size):
        kernel[i] = np.exp(-((i - mid) / sigma) ** 2 / 2)
        dkernel[i] = -(i - mid) / (np.sqrt(2 * np.pi) * sigma ** 3) * kernel[i]
    kernel = kernel / kernel.sum()
    return kernel, dkernel


def img_padding(img, padding=0, padding_mode='zeros'):
    if isinstance(padding, int):
        padding = (0, 0)

    pad_width = [[0, 0] for _ in range(img.ndim)]
    pad_width[-2] = [padding[0], padding[0]]
    pad_width[-1] = [padding[1], padding[1]]
    if padding_mode == 'zeros':
        return np.pad(img, pad_width)
    elif padding_mode == 'reflect':
        return np.pad(img, pad_width, mode='reflect')
    elif padding_mode == 'replicate':
        return np.pad(img, pad_width, mode='symmetric')
    else:
        return NotImplemented


def convolution(img, kernel, padding_mode='zeros'):
    '''
    img: C*H*W, where C is channel (3 for rgb image, 1 for grayscale);
                  H and W are image height and width respectively.
    '''
    assert padding_mode in {'zeros', 'reflect', 'replicate'}, 'wrong padding mode'
    kernel_size = kernel.shape
    h_k, w_k = kernel.shape  # kernel height and width
    padding = (h_k // 2, w_k // 2)

    if isinstance(padding, int):
        padding = (padding, padding)

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    padded = img_padding(img, padding, padding_mode)

    C, H_in, W_in = img.shape  # batch size, rgb channel, Height, Width
    H_out = np.floor(H_in + 2 * padding[0] - kernel_size[0] + 1).astype(int)
    W_out = np.floor(W_in + 2 * padding[1] - kernel_size[1] + 1).astype(int)

    expanded = np.lib.stride_tricks.as_strided(
        padded,
        shape=(
            H_out,  # out channel height
            W_out,  # out channel width
            padded.shape[-3],  # input channel
            kernel.shape[-2],  # kernel height
            kernel.shape[-1],  # kernel width
        ),
        strides=(
            padded.strides[-2],  # H dimension
            padded.strides[-1],  # W dimension
            padded.strides[-3],  # input chennel
            padded.strides[-2],  # kernel height
            padded.strides[-1],  # kernel width
        ),
        writeable=False,
    )
    feature_map = np.ascontiguousarray(np.moveaxis(np.einsum('...ij,...ij->...', expanded, kernel), -1, -3))
    return feature_map


@jit(nopython=True, cache=True)
def non_max_suppression(gradient_mag, gradient_dir,i=1):
    output = np.zeros_like(gradient_mag)
    h, w = gradient_mag.shape[:2]
    for row in range(1, h - 1):
        for col in range(1, w - 1):
            rh = min(h-1,row+i)
            rl = max(0, row - i)
            ch = min(w-1, col + i)
            cl = max(0, col - i)
            if (1 / 8 * np.pi < gradient_dir[row, col] <= 3 / 8 * np.pi) or (
                    -7 / 8 * np.pi < gradient_dir[row, col] <= -5 / 8 * np.pi):
                p1 = gradient_mag[rh, cl]
                p2 = gradient_mag[rl, ch]
            elif (3 / 8 * np.pi < gradient_dir[row, col] <= 5 / 8 * np.pi) or (
                    -5 / 8 * np.pi < gradient_dir[row, col] <= -3 / 8 * np.pi):
                p1 = gradient_mag[rh, col]
                p2 = gradient_mag[rl, col]
            elif (5 / 8 * np.pi < gradient_dir[row, col] <= 7 / 8 * np.pi) or (
                    -3 / 8 * np.pi < gradient_dir[row, col] <= -1 / 8 * np.pi):
                p1 = gradient_mag[rl, cl]
                p2 = gradient_mag[rh, ch]
            else:
                p1 = gradient_mag[row, cl]
                p2 = gradient_mag[row, ch]
            if gradient_mag[row, col] >= p1 and gradient_mag[row, col] >= p2:
                output[row, col] = gradient_mag[row, col]
    return output


def double_thres(img, low=None, high=None):
    output = np.zeros_like(img)
    if high is None:
        low = 0.1
        high = 0.2
    output[img >= high] = 1  # white
    output[(img >= low) & (img < high)] = 0.5  # gray
    strong = deque([(i[0], i[1]) for i in np.transpose((img >= high).nonzero())])
    return output, strong


def hysteresis(img_thres, strong):
    img = img_thres.copy()
    while len(strong) != 0:
        popped = strong.popleft()
        row, col = popped
        check = ((row - 1, col - 1), (row - 1, col), (row - 1, col + 1),
                 (row, col - 1), (row, col + 1),
                 (row + 1, col - 1), (row + 1, col), (row + 1, col + 1))
        for i in check:
            if img[i] == 0.5:
                img[i] = 1
                strong.append(i)
    img[img == 0.5] = 0
    output = np.zeros_like(img, dtype=bool)
    output[img == 1] = True
    return output


def myCannyEdgeDetector(img, low, high,line_width=1):
    # change image to grayscale
    if len(img.shape) == 3:
        img = rgb2gray(img)
    else:
        img = img / 255
    kernel, d_kernel = get_gaussian_kernel(sigma=1)

    img = img[None, ...]

    # calculate gradient (combine gaussian blur and derivative)
    # don't forget the negative, because coordinate is different
    img_x = -convolution(convolution(img, kernel), np.transpose(d_kernel))
    img_y = convolution(convolution(img, d_kernel), np.transpose(kernel))

    img_x = img_x[0]
    img_y = img_y[0]

    # gradient magnitude and direction
    gradient_mag = np.sqrt(img_x ** 2 + img_y ** 2)
    # direction
    gradient_dir = np.arctan2(img_y, img_x)

    # non-max suppression
    gradient_sup = non_max_suppression(gradient_mag, gradient_dir,i=line_width)
    gradient_sup = gradient_sup / np.max(gradient_sup)  # normalize so max is 1

    # double threshold and hysteresis
    img_thres, strong = double_thres(gradient_sup, low, high)
    my_edge = hysteresis(img_thres, strong)
    return my_edge


if __name__ == '__main__':
    # read image
    image = io.imread('images/lena.jpg')
    # run canny edge detection
    my_edge = myCannyEdgeDetector(image, 0.1, 0.2)
