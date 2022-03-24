import matplotlib.image as image
import numpy as np
import os
from skimage.transform import resize

# importing and downsizing images to 100x100 px


def import_images(resolution):
    green_images = import_green(resolution)
    ripe_images = import_ripe(resolution)
    overripe_images = import_overripe(resolution)
    test_samples_img = import_test_samples(resolution)
    images = np.concatenate((green_images, ripe_images, overripe_images))
    cat_green = np.full(len(green_images), 'g')
    cat_ripe = np.full(len(ripe_images), 'r')
    cat_overripe = np.full(len(overripe_images), 'o')
    categories = np.concatenate((cat_green, cat_ripe, cat_overripe))
    return [categories, images, test_samples_img]


# import photos of green bananas
def import_green(resolution):
    dirlist_green = os.listdir('./pictures/learning/green')  # list of all file names in directory
    green_img = np.empty((len(dirlist_green), resolution, resolution, 3), dtype=float)  # empty array for downsized color photos
    for i, im_name in enumerate(dirlist_green):
        print(i, '. green')  # print current iteration
        img = image.imread(f'./pictures/learning/green/{im_name}')  # read a photo
        green_img[i] = resize(img, (resolution, resolution), anti_aliasing=True)  # downsize a photo to 100x100 px
    return green_img


# import photos of ripe bananas
def import_ripe(resolution):
    dirlist_ripe = os.listdir('./pictures/learning/ripe')
    ripe_img = np.empty((len(dirlist_ripe), resolution, resolution, 3), dtype=float)
    for i, im_name in enumerate(dirlist_ripe):
        print(i, '. ripe')
        img = image.imread(f'./pictures/learning/ripe/{im_name}')
        ripe_img[i] = resize(img, (resolution, resolution), anti_aliasing=True)
    return ripe_img


# import photos of overripe bananas
def import_overripe(resolution):
    dirlist_overripe = os.listdir('./pictures/learning/overripe')
    overripe_img = np.empty((len(dirlist_overripe), resolution, resolution, 3), dtype=float)
    for i, im_name in enumerate(dirlist_overripe):
        print(i, '. overripe')
        img = image.imread(f'./pictures/learning/overripe/{im_name}')
        overripe_img[i] = resize(img, (resolution, resolution), anti_aliasing=True)
    return overripe_img


# import test samples
def import_test_samples(resolution):
    dirlist_overripe = os.listdir('./pictures/test')
    test_samples = np.empty((len(dirlist_overripe), resolution, resolution, 3), dtype=float)
    for i, im_name in enumerate(dirlist_overripe):
        print(i, '. test sample')
        img = image.imread(f'./pictures/test/{im_name}')
        test_samples[i] = resize(img, (resolution, resolution), anti_aliasing=True)
    return test_samples
