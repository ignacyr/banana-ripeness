import time as t

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report


class BananaClassifier:
    """Banana ripeness classification based on the photos."""
    def __init__(self, classifier, images: np.ndarray, categories: np.ndarray, resolution: int):
        """Constructor initializes classifier object."""
        self.classifier = classifier
        self.learning_time = 0.0  # there should be a better way
        self.images = images
        self.categories = categories
        self.resolution = resolution
        self.reshaped_images = self.images.reshape(len(self.images), 3 * self.resolution ** 2)
        self.predictions = np.array([])
        self.reshaped_test_samples = np.array([])
        self.test_samples = np.array([])
        return

    def fit(self):
        """Learning classifier to classify bananas based on photos."""
        start_time = t.time()

        if type(self.classifier) != tuple:  # If not tuple learn single classifier.
            self.classifier.fit(self.reshaped_images, self.categories)
        else:  # If tuple learn each classifier in a for loop.
            for clf in self.classifier:
                clf.fit(self.reshaped_images, self.categories)

        end_time = t.time()
        self.learning_time = end_time - start_time
        return

    def predict(self, test_samples: np.ndarray, resolution: int):
        self.test_samples = test_samples
        self.reshaped_test_samples = test_samples.reshape(len(test_samples), 3 * resolution ** 2)
        for i in range(len(self.reshaped_test_samples)):
            if type(self.classifier) == tuple:
                all_probas = [None] * len(self.classifier)
                for y in range(len(self.classifier)):
                    all_probas[y] = self.classifier[y].predict_proba(
                        self.reshaped_test_samples[i].reshape(1, -1)
                    )
                avg = (sum(all_probas) / len(all_probas))[0]
                if avg[0] == max(avg):
                    self.predictions = np.append(self.predictions, 'green')
                elif avg[1] == max(avg):
                    self.predictions = np.append(self.predictions, 'overripe')
                else:
                    self.predictions = np.append(self.predictions, 'ripe')
            else:
                self.predictions[i] = self.classifier.predict(self.reshaped_test_samples[i].reshape(1, -1))[0]
        return self.predictions

    def plot(self, title=""):
        n_rows = 3
        n_cols = 5
        _, axes = plt.subplots(n_rows, n_cols)
        for i in range(n_rows):
            for j in range(n_cols):
                samples_index = i * n_cols + j
                print(self.reshaped_test_samples[samples_index].shape)  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                axes[i][j].imshow(self.test_samples[samples_index])
                axes[i][j].axis('off')
                axes[i][j].set_title(self.predictions[samples_index])
        if title:
            plt.suptitle(title)
        else:
            classifiers_str = ""
            if type(self.classifier) == tuple:
                for obj in self.classifier:
                    if classifiers_str:
                        classifiers_str = classifiers_str + ', ' + obj.__str__()
                    else:
                        classifiers_str = obj.__str__()
            else:
                classifiers_str = self.classifier.__str__()
            plt.suptitle(f"Learning time of {classifiers_str}: {round(self.learning_time, 3)} [s]")
        plt.show()
        return

    def predict_and_plot(self, test_samples: np.ndarray, resolution: int, title=""):
        self.predict(test_samples, resolution)
        self.plot()
        return

    def report(self):
        """Display classification report"""
        if type(self.classifier) == tuple:
            for clf in self.classifier:
                print(clf.__str__())
                print(classification_report(self.categories, clf.predict(self.reshaped_images)))
        else:
            print(self.classifier.__str__())
            print(classification_report(self.categories, self.classifier.predict(self.reshaped_images)))
        return
