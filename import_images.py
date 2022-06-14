import os

import matplotlib.image as image
import numpy as np
from skimage.transform import resize


def import_directory(resolution, path: str):
    """Importing and downsizing images to 100x100 px"""
    dirlist = os.listdir(path)
    images_array = np.empty((len(dirlist), resolution, resolution, 3), dtype=float)
    for i, im_name in enumerate(dirlist):
        print(i, f'. {path}')
        img = image.imread(f'{path}/{im_name}')
        images_array[i] = resize(img, (resolution, resolution), anti_aliasing=True)
    return images_array


def save_images(resolution):
    """Save images and categories to .npy files"""
    green_images = import_directory(resolution, './pictures/learning/green')
    ripe_images = import_directory(resolution, './pictures/learning/ripe')
    overripe_images = import_directory(resolution, './pictures/learning/overripe')
    test_samples_img = import_directory(resolution, './pictures/testing')

    images = np.concatenate((green_images, ripe_images, overripe_images))

    cat_green = np.full(len(green_images), 'green')
    cat_ripe = np.full(len(ripe_images), 'ripe')
    cat_overripe = np.full(len(overripe_images), 'overripe')
    categories = np.concatenate((cat_green, cat_ripe, cat_overripe))

    np.save('data/categories', categories)
    np.save('data/images', images)
    np.save('data/test_samples_img', test_samples_img)
    return [categories, images, test_samples_img]
