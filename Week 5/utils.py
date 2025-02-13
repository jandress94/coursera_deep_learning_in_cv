# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/utils.py
#   + License: MIT
from __future__ import division
import math
import random
import scipy.misc
import numpy as np
from skimage.transform import resize

def get_image(image_path, image_size, is_crop=True, img_size = 64):
    return transform(imread(image_path), image_size, is_crop)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path):
    return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((int(h * size[0]), int(w * size[1]), 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img

def imsave(images, size, path):
    img = merge(images, size)
    return scipy.misc.imsave(path, (255*img).astype(np.uint8))

def transform(image, npx=64, is_crop=True):
    i = np.array(image)/127.5 - 1.
    
    if is_crop:
        min_d = np.min(i.shape[:2])
        d0 = (i.shape[0] - min_d) // 2
        d1 = (i.shape[1] - min_d) // 2
        
        i = i[d0 : d0 + min_d, d1 : d1 + min_d, :]
    
    return resize(i, (npx, npx))

def inverse_transform(images):
    return (images+1.)/2.
