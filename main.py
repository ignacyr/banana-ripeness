from import_images import import_images
from skimage.color import rgb2gray
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from skimage.feature import hog


def gray_images_mlp_classification():
    gray_images = rgb2gray(images)
    [samples, nx, ny] = gray_images.shape
    gray_images_2d = gray_images.reshape((samples, nx * ny))

    gray_test_samples = rgb2gray(test_samples)
    [n_test_samples, nx_s, ny_s] = gray_test_samples.shape
    test_samples_2d = gray_test_samples.reshape((n_test_samples, nx_s * ny_s))

    mlp_class = MLPClassifier()

    mlp_class.fit(gray_images_2d, categories)  # learning

    n_rows = 2
    n_cols = 3
    _, axes = plt.subplots(n_rows, n_cols)
    for i in range(n_rows):
        for j in range(n_cols):
            samples_index = i * 3 + j
            axes[i][j].imshow(test_samples[samples_index])
            axes[i][j].axis('off')
            axes[i][j].set_title(f'MLP Label: {mlp_class.predict(test_samples_2d[samples_index].reshape(1, -1))}')
    plt.show()
    # print(classification_report(categories, mlp_class.predict(gray_images_2d)))
    return


def color_images_mlp_classification():
    images_hog = hog(
        images[12],
        cells_per_block=(2, 2),
        orientations=9,
        visualize=True,
        channel_axis=-1,
        block_norm='L2-Hys')

    mlp_class = MLPClassifier()

    mlp_class.fit(images_hog, categories)  # learning

    n_rows = 2
    n_cols = 3
    _, axes = plt.subplots(n_rows, n_cols)
    for i in range(n_rows):
        for j in range(n_cols):
            samples_index = i * 3 + j
            axes[i][j].imshow(test_samples[samples_index])
            axes[i][j].axis('off')
            axes[i][j].set_title(f'MLP Label: {mlp_class.predict(images_hog[samples_index].reshape(1, -1))}')
    plt.show()
    return


[categories, images, test_samples] = import_images()

gray_images_mlp_classification()
color_images_mlp_classification()
