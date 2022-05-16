import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier

import classification_class as cc
import import_images as ii


def main():
    resolution = 100  # don't set to more than 100 -

    # ii.save_images(resolution)  # comment out if new pictures have been added

    categories = np.load('data/categories.npy')
    images = np.load('data/images.npy')
    test_samples = np.load('data/test_samples_img.npy')

    classifier = cc.BananaClassifier((RandomForestClassifier(), LinearDiscriminantAnalysis()),
                                     images, categories, resolution)
    classifier.fit()
    classifier.predict_and_plot(test_samples, resolution, "MIX")
    classifier.report()


if __name__ == '__main__':
    main()
