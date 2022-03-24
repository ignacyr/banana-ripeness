from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np


def mlp_classification(categories, images, test_samples):
    np.random.shuffle(test_samples)
    reshaped_images = images.reshape(248, 30000)
    reshaped_test_samples = test_samples.reshape(15, 30000)

    mlp_class = MLPClassifier(max_iter=1000)
    mlp_class.fit(reshaped_images, categories)  # learning

    n_rows = 3
    n_cols = 5
    _, axes = plt.subplots(n_rows, n_cols)
    for i in range(n_rows):
        for j in range(n_cols):
            samples_index = i * n_cols + j
            axes[i][j].imshow(test_samples[samples_index])
            axes[i][j].axis('off')
            axes[i][j].set_title(f'MLP Label: {mlp_class.predict(reshaped_test_samples[samples_index].reshape(1, -1))}')
    plt.show()
