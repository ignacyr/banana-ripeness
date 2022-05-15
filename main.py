import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier

import classification as clf
import import_images as ii


def main():
    resolution = 100  # don't set to more than 100
    # ii.save_images(resolution)  # comment out if new pictures have been added

    categories = np.load('data/categories.npy')
    images = np.load('data/images.npy')
    test_samples = np.load('data/test_samples_img.npy')

    clf.classification(categories, images, test_samples, resolution, (RandomForestClassifier(),
                       LinearDiscriminantAnalysis(), MLPClassifier()))

    clf.classification(categories, images, test_samples, resolution, RandomForestClassifier())
    clf.classification(categories, images, test_samples, resolution, LinearDiscriminantAnalysis())
    clf.classification(categories, images, test_samples, resolution, MLPClassifier())


if __name__ == '__main__':
    main()
