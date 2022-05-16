import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier

from classification import BananaClassifier
import import_images as ii


def main():
    resolution = 100  # don't set to more than 100 -

    # ii.save_images(resolution)  # comment out if new pictures have been added

    categories = np.load('data/categories.npy')
    images = np.load('data/images.npy')
    test_samples = np.load('data/test_samples_img.npy')

    classifier = BananaClassifier(LinearDiscriminantAnalysis())
    classifier.fit(images, categories, resolution)
    classifier.predict_and_plot(test_samples, resolution)

    classifiers = BananaClassifier(LinearDiscriminantAnalysis(), RandomForestClassifier(), MLPClassifier())
    classifiers.fit(images, categories, resolution)
    classifiers.predict_and_plot(test_samples, resolution)


if __name__ == '__main__':
    main()
