import matplotlib.image as image
import numpy as np
import os
from skimage.transform import resize
from tempfile import TemporaryFile


# importing and downsizing images to 100x100 px
def import_directory(resolution, path: str):
    dirlist = os.listdir(path)
    images_array = np.empty((len(dirlist), resolution, resolution, 3), dtype=float)
    for i, im_name in enumerate(dirlist):
        print(i, f'. {path}')
        img = image.imread(f'{path}/{im_name}')
        images_array[i] = resize(img, (resolution, resolution), anti_aliasing=True)
    return images_array


# save images and categories to .npy files
def save_images(resolution):
    green_images = import_directory(resolution, './pictures/learning/green')
    ripe_images = import_directory(resolution, './pictures/learning/ripe')
    overripe_images = import_directory(resolution, './pictures/learning/overripe')
    test_samples_img = import_directory(resolution, './pictures/test')

    images = np.concatenate((green_images, ripe_images, overripe_images))

    cat_green = np.full(len(green_images), 'green')
    cat_ripe = np.full(len(ripe_images), 'ripe')
    cat_overripe = np.full(len(overripe_images), 'overripe')
    categories = np.concatenate((cat_green, cat_ripe, cat_overripe))

    np.save('data/categories', categories)
    np.save('data/images', images)
    np.save('data/test_samples_img', test_samples_img)
    return [categories, images, test_samples_img]
