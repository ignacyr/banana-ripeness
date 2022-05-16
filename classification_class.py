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
        self.reshaped_images = self.images.reshape(248, 3 * self.resolution ** 2)
        return

    def fit(self):
        """Learning classifier to classify bananas based on photos."""
        start_time = t.time()

        if type(self.classifier) != tuple:  # If not tuple learn single object.
            self.classifier.fit(self.reshaped_images, self.categories)
        else:  # If tuple learn each object in for loop.
            for i in range(len(self.classifier)):
                self.classifier[i].fit(self.reshaped_images, self.categories)

        end_time = t.time()
        self.learning_time = end_time - start_time
        return

    def predict(self, test_samples: np.ndarray, resolution: int, title=""):  # fix cohesion
        """Predict banana ripeness based on the photos."""
        reshaped_test_samples = test_samples.reshape(15, 3 * resolution ** 2)

        n_rows = 3
        n_cols = 5
        _, axes = plt.subplots(n_rows, n_cols)

        for i in range(n_rows):
            for j in range(n_cols):
                samples_index = i * n_cols + j
                axes[i][j].imshow(test_samples[samples_index])
                axes[i][j].axis('off')
                if type(self.classifier) != tuple:
                    axes[i][j].set_title(self.classifier.predict(reshaped_test_samples[samples_index].reshape(1, -1))[0])
                else:
                    all_probas = [None] * len(self.classifier)
                    for y in range(len(self.classifier)):
                        all_probas[y] = self.classifier[y].predict_proba(reshaped_test_samples[samples_index].reshape(1, -1))
                    avg = sum(all_probas) / len(all_probas)
                    avg = avg[0]
                    if avg[0] == max(avg):
                        axes[i][j].set_title('green')
                    elif avg[1] == max(avg):
                        axes[i][j].set_title('overripe')
                    else:
                        axes[i][j].set_title('ripe')
        classifiers_str = ""
        if type(self.classifier) == tuple:
            for obj in self.classifier:
                if len(classifiers_str):
                    classifiers_str = classifiers_str + ', ' + obj.__str__()
                else:
                    classifiers_str = obj.__str__()
        else:
            classifiers_str = self.classifier.__str__()

        if title:
            plt.suptitle(title)
        else:
            plt.suptitle(f"Learning time of {classifiers_str}: {round(self.learning_time, 3)} [s]")
        plt.show()
        return

    def plot(self):
        return

    def predict_and_plot(self):
        return

    def report(self):
        """Display classification report"""
        print(classification_report(self.categories, self.classifier.predict(self.reshaped_images)))
        return
