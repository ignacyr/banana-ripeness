import time as t

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report


def classification(categories, images, test_samples, resolution, classifier):
    """
    A universal function for classification.
    Pass an object of a classifier class as an argument 'classifier'.
    """
    # np.random.shuffle(test_samples)
    reshaped_images = images.reshape(248, 3 * resolution**2)
    reshaped_test_samples = test_samples.reshape(15, 3 * resolution**2)

    start_time = t.time()
    classifier.fit(reshaped_images, categories)  # learning
    end_time = t.time()
    learning_time = end_time - start_time
    print(f"Learning time of {classifier.__str__()}: {t.strftime('%M:%S', t.gmtime(learning_time))} [mm:ss]")

    print(classifier.__str__())

    n_rows = 3
    n_cols = 5
    _, axes = plt.subplots(n_rows, n_cols)

    for i in range(n_rows):
        for j in range(n_cols):
            samples_index = i * n_cols + j
            axes[i][j].imshow(test_samples[samples_index])
            axes[i][j].axis('off')
            axes[i][j].set_title(classifier.predict(reshaped_test_samples[samples_index].reshape(1, -1)))
            print(classifier.predict_proba(reshaped_test_samples[samples_index].reshape(1, -1)))

    plt.suptitle(classifier.__str__())  # a name of a classifier
    plt.show()

    # print(classification_report(categories, classifier.predict(reshaped_images)))  # a classification report
    return
