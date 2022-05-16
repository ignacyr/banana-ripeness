import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from classification import BananaClassifier, Metrics
import import_images as ii


def main():
    resolution = 100  # don't set to more than 100 -

    # ii.save_images(resolution)  # comment out if new pictures have been added

    categories = np.load('data/categories.npy')
    images = np.load('data/images.npy')
    test_samples = np.load('data/test_samples_img.npy')

    classifier1 = BananaClassifier(LinearDiscriminantAnalysis(), RandomForestClassifier(), DecisionTreeClassifier())
    classifier1.fit(images, categories, resolution)
    classifier1.predict_and_plot(test_samples, resolution)
    metrics1 = Metrics(classifier1)
    metrics1.accuracy()
    # metrics1.report()

    classifier2 = BananaClassifier(MLPClassifier(max_iter=1000))
    classifier2.fit(images, categories, resolution)
    classifier2.predict_and_plot(test_samples, resolution)
    metrics2 = Metrics(classifier2)
    metrics2.accuracy()
    # metrics2.report()


if __name__ == '__main__':
    main()
