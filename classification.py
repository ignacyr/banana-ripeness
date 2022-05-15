import time as t

import matplotlib.pyplot as plt
import numpy as np
# from sklearn.metrics import classification_report


def classification(categories: np.ndarray, images: np.ndarray, test_samples: np.ndarray,
                   resolution: int, classifier, title=""):
    """
    A universal function for classification.
    Pass an object of a classifier class as an argument 'classifier'.
    """
    reshaped_images = images.reshape(248, 3 * resolution**2)
    reshaped_test_samples = test_samples.reshape(15, 3 * resolution**2)

    start_time = t.time()
    if type(classifier) != tuple:
        classifier.fit(reshaped_images, categories)  # learning
    else:
        for i in range(len(classifier)):
            classifier[i].fit(reshaped_images, categories)  # learning
    end_time = t.time()
    learning_time = end_time - start_time

    n_rows = 3
    n_cols = 5
    _, axes = plt.subplots(n_rows, n_cols)

    for i in range(n_rows):
        for j in range(n_cols):
            samples_index = i * n_cols + j
            axes[i][j].imshow(test_samples[samples_index])
            axes[i][j].axis('off')
            if type(classifier) != tuple:
                axes[i][j].set_title(classifier.predict(reshaped_test_samples[samples_index].reshape(1, -1))[0])
            else:
                all_probas = [None] * len(classifier)
                for y in range(len(classifier)):
                    all_probas[y] = classifier[y].predict_proba(reshaped_test_samples[samples_index].reshape(1, -1))
                avg = sum(all_probas) / len(all_probas)
                avg = avg[0]

                if avg[0] == max(avg):
                    axes[i][j].set_title('green')
                elif avg[1] == max(avg):
                    axes[i][j].set_title('overripe')
                else:
                    axes[i][j].set_title('ripe')

    classifiers_str = ""
    if type(classifier) == tuple:
        for obj in classifier:
            if len(classifiers_str):
                classifiers_str = classifiers_str + ', ' + obj.__str__()
            else:
                classifiers_str = obj.__str__()
    else:
        classifiers_str = classifier.__str__()

    if title:
        plt.suptitle(title)
    else:
        plt.suptitle(f"Learning time of {classifiers_str}: {round(learning_time, 3)} [s]")
    plt.show()

    # print(classification_report(categories, classifier.predict(reshaped_images)))
    return
