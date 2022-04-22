import import_images as ii
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import classification as clf


def main():
    resolution = 100  # don't set to more than 100
    # ii.save_images(resolution)  # comment out if new pictures have been added

    categories = np.load('data/categories.npy')
    images = np.load('data/images.npy')
    test_samples = np.load('data/test_samples_img.npy')

    clf.classification(categories, images, test_samples, resolution, DecisionTreeClassifier())
    # clf.classification(categories, images, test_samples, resolution, SVC(max_iter=1000))
    # clf.classification(categories, images, test_samples, resolution, MLPClassifier(max_iter=200))


if __name__ == '__main__':
    main()
