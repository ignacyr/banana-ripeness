from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np


def decision_tree_classification(categories, images, test_samples, resolution):
    np.random.shuffle(test_samples)
    reshaped_images = images.reshape(248, 3 * resolution**2)
    reshaped_test_samples = test_samples.reshape(15, 3 * resolution**2)

    dct = DecisionTreeClassifier()
    dct.fit(reshaped_images, categories)  # learning

    n_rows = 3
    n_cols = 5
    _, axes = plt.subplots(n_rows, n_cols)
    for i in range(n_rows):
        for j in range(n_cols):
            samples_index = i * n_cols + j
            axes[i][j].imshow(test_samples[samples_index])
            axes[i][j].axis('off')
            axes[i][j].set_title(dct.predict(reshaped_test_samples[samples_index].reshape(1, -1)))
    plt.suptitle('Decision Tree')
    plt.show()

    print(classification_report(categories, dct.predict(reshaped_images)))
    return
