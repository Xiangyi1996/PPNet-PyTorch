# import the necessary packages
from skimage import io
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
import os.path as osp
import cv2
import json
from skimage.segmentation import slic
from skimage.segmentation import *
from skimage.util import img_as_float
from skimage import io
import argparse
import os
import torch
import copy
from PIL import Image
import tqdm
import time

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def create_image_with_mask(img, mask):
    mask[mask>0] = 1
    # print(img.shape)
    zero_pad = np.zeros_like(mask)
    green_pad = np.stack((zero_pad, mask, zero_pad), axis=2) * 255 * 0.5

    img[mask == 1] = img[mask == 1] * 0.5
    final_img = (img + green_pad).astype(np.uint8)

    return final_img


def show_superpixel(image, mask, segments):
    # plt.subplot(1, 3, 1)
    # plt.imshow(image)

    plt.subplot(1, 2, 1)
    gray_img = create_image_with_mask(image, mask)
    plt.imshow(gray_img)
    plt.axis("off")

    image = img_as_float(image)

    plt.subplot(1, 2, 2)
    plt.imshow(mark_boundaries(image, segments))

    plt.savefig('1.png')

