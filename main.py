from import_images import import_images
import numpy as np
from sklearn import svm
from skimage.color import rgb2gray

[categories, images] = import_images()

gray_images = rgb2gray(images)
