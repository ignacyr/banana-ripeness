import matplotlib.image as image
import numpy as np
import os
from skimage.transform import resize


def import_green():
    dirlist_green = os.listdir('./pictures/learning/green')
    green_img = np.empty((len(dirlist_green), 100, 100, 3), dtype=float)
    for i, im_name in enumerate(dirlist_green):
        print(i, '. green')
        img = image.imread(f'./pictures/learning/green/{im_name}')
        img_resized = resize(img, (100, 100), anti_aliasing=True)
        green_img[i] = img_resized


def import_ripe():
    dirlist_ripe = os.listdir('./pictures/learning/ripe')
    ripe_img = np.empty((len(dirlist_ripe), 100, 100, 3), dtype=float)
    for i, im_name in enumerate(dirlist_ripe):
        print(i, '. ripe')
        img = image.imread(f'./pictures/learning/ripe/{im_name}')
        img_resized = resize(img, (100, 100), anti_aliasing=True)
        ripe_img[i] = img_resized


def import_overripe():
    dirlist_overripe = os.listdir('./pictures/learning/overripe')
    overripe_img = np.empty((len(dirlist_overripe), 100, 100, 3), dtype=float)
    for i, im_name in enumerate(dirlist_overripe):
        print(i, '. overripe')
        img = image.imread(f'./pictures/learning/overripe/{im_name}')
        img_resized = resize(img, (100, 100), anti_aliasing=True)
        overripe_img[i] = img_resized
