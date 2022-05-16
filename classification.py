import time as t

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, accuracy_score


class BananaClassifier:
    """Banana ripeness classification based on the photos."""
    def __init__(self, *classifier):
        """Constructor initializes classifier object."""
        self.classifier = classifier
        self.learning_time = 0.0  # there should be a better way
        self.predicting_time = 0.0
        self.images = np.array([])
        self.categories = np.array([])
        self.resolution = np.array([])
        self.reshaped_images = np.array([])
        self.predictions = np.array([])
        self.reshaped_test_samples = np.array([])
        self.test_samples = np.array([])
        return

    def fit(self, images: np.ndarray, categories: np.ndarray, resolution: int):
        """Learn classifier to classify bananas based on photos."""
        self.images = images
        self.categories = categories
        self.resolution = resolution
        self.reshaped_images = self.images.reshape(len(self.images), 3 * self.resolution ** 2)

        start_time = t.time()

        for clf in self.classifier:
            clf.fit(self.reshaped_images, self.categories)

        end_time = t.time()
        self.learning_time = end_time - start_time
        return

    def predict(self, test_samples: np.ndarray, resolution: int):
        """Predict if banana is green, ripe or overripe."""
        self.test_samples = test_samples
        self.reshaped_test_samples = test_samples.reshape(len(test_samples), 3 * resolution ** 2)
        for i, _ in enumerate(self.reshaped_test_samples):
            all_probas = [None] * len(self.classifier)

            start_time = t.time()

            for y, _ in enumerate(self.classifier):
                all_probas[y] = self.classifier[y].predict_proba(self.reshaped_test_samples[i].reshape(1, -1))

            end_time = t.time()
            self.predicting_time = end_time - start_time

            avg = (sum(all_probas) / len(all_probas))[0]  # getting array out of array
            if avg[0] == max(avg):
                self.predictions = np.append(self.predictions, 'green')
            elif avg[1] == max(avg):
                self.predictions = np.append(self.predictions, 'overripe')
            else:
                self.predictions = np.append(self.predictions, 'ripe')
        return self.predictions

    def plot(self, title=""):
        """Display categorized photos of the bananas."""
        n_rows = 3
        n_cols = 5
        _, axes = plt.subplots(n_rows, n_cols)
        for i in range(n_rows):
            for j in range(n_cols):
                samples_index = i * n_cols + j
                axes[i][j].imshow(self.test_samples[samples_index])
                axes[i][j].axis('off')
                axes[i][j].set_title(self.predictions[samples_index])
        if title:
            plt.suptitle(title)
        else:
            classifiers_str = ""
            for obj in self.classifier:
                if classifiers_str:
                    classifiers_str = f"{classifiers_str}, {obj.__str__()}"
                else:
                    classifiers_str = obj.__str__()
            plt.suptitle(f"{classifiers_str}\n "
                         f"Learning time: {round(self.learning_time, 3)} [s]\n"
                         f"Predicting time: {round(1000 * self.predicting_time, 3)} [ms]")
        plt.show()
        return

    def predict_and_plot(self, test_samples: np.ndarray, resolution: int, title=""):
        """Run predict and plot functions one after another."""
        self.predict(test_samples, resolution)
        self.plot(title)
        return self.predictions


class Metrics:
    def __init__(self, b_classifier: BananaClassifier):
        self.b_classifier = b_classifier
        pass

    def report(self):
        """Display classification report"""
        for clf in self.b_classifier.classifier:
            print(clf.__str__())
            print(classification_report(self.b_classifier.categories, clf.predict(self.b_classifier.reshaped_images)))
        return

    def accuracy(self):
        y_true = np.array(['overripe', 'ripe', 'green', 'ripe', 'overripe', 'overripe', 'ripe', 'green',
                           'green', 'ripe', 'ripe', 'overripe', 'green', 'green', 'overripe'])  # change to auto
        y_pred = self.b_classifier.predictions
        print(accuracy_score(y_true, y_pred))
        return


