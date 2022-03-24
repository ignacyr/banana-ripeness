from import_images import import_images
from MLP_classification import mlp_classification


resolution = 64  # don't set to more than 100
[categories, images, test_samples] = import_images(resolution)

mlp_classification(categories, images, test_samples, resolution)
