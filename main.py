from import_images import import_images
from skimage.color import rgb2gray
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

[categories, images, test_samples] = import_images()

gray_images = rgb2gray(images)
[samples, nx, ny] = gray_images.shape
gray_images_2d = gray_images.reshape((samples, nx * ny))

gray_test_samples = rgb2gray(test_samples)
[n_test_samples, nx_s, ny_s] = gray_test_samples.shape
test_samples_2d = gray_test_samples.reshape((n_test_samples, nx_s * ny_s))

mlp_class = MLPClassifier()

mlp_class.fit(gray_images_2d, categories)

categories_predicted = mlp_class.predict(gray_images_2d)

n_rows = 2
n_cols = 3
_, axes = plt.subplots(n_rows, n_cols)
plt.gray()
for i in range(n_rows):
    for j in range(n_cols):
        samples_index = i * 3 + j
        axes[i][j].imshow(test_samples[samples_index])
        axes[i][j].axis('off')
        axes[i][j].set_title(f'MLP Label: {mlp_class.predict(test_samples_2d[samples_index].reshape(1, -1))}')

plt.show()
print(classification_report(categories, mlp_class.predict(gray_images_2d)))
