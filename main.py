import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

from classification import BananaClassifier
import import_images as ii


def main():
    resolution = 100  # don't set to more than 100 -

    # ii.save_images(resolution)  # comment out if new pictures have been added

    categories = np.load('data/categories.npy')
    images = np.load('data/images.npy')
    test_samples = np.load('data/test_samples_img.npy')

    classifier1 = BananaClassifier(LinearDiscriminantAnalysis(), RandomForestClassifier())
    # classifier1.fit(images, categories, resolution)
    # classifier1.predict_and_plot(test_samples, resolution)
    # classifier1.print_metrics()
    classifier1.fit(images, categories, resolution)
    classifier1.learning_curve(images, categories, resolution, 8)
    return


if __name__ == '__main__':
    main()
