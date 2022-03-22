import matplotlib.image as image
import numpy as np
import os
from skimage.transform import resize


def import_images():
    green_images = import_green()
    ripe_images = import_ripe()
    overripe_images = import_overripe()
    images = np.concatenate((green_images, ripe_images, overripe_images))
    cat_green = np.full(len(green_images), 'g')
    cat_ripe = np.full(len(ripe_images), 'r')
    cat_overripe = np.full(len(overripe_images), 'o')
    categories = np.concatenate((cat_green, cat_ripe, cat_overripe))
    return [categories, images]


def import_green():
    dirlist_green = os.listdir('./pictures/learning/green')
    green_img = np.empty((len(dirlist_green), 100, 100, 3), dtype=float)
    for i, im_name in enumerate(dirlist_green):
        print(i, '. green')
        img = image.imread(f'./pictures/learning/green/{im_name}')
        img_resized = resize(img, (100, 100), anti_aliasing=True)
        green_img[i] = img_resized
    return green_img


def import_ripe():
    dirlist_ripe = os.listdir('./pictures/learning/ripe')
    ripe_img = np.empty((len(dirlist_ripe), 100, 100, 3), dtype=float)
    for i, im_name in enumerate(dirlist_ripe):
        print(i, '. ripe')
        img = image.imread(f'./pictures/learning/ripe/{im_name}')
        img_resized = resize(img, (100, 100), anti_aliasing=True)
        ripe_img[i] = img_resized
    return ripe_img


def import_overripe():
    dirlist_overripe = os.listdir('./pictures/learning/overripe')
    overripe_img = np.empty((len(dirlist_overripe), 100, 100, 3), dtype=float)
    for i, im_name in enumerate(dirlist_overripe):
        print(i, '. overripe')
        img = image.imread(f'./pictures/learning/overripe/{im_name}')
        img_resized = resize(img, (100, 100), anti_aliasing=True)
        overripe_img[i] = img_resized
    return overripe_img
