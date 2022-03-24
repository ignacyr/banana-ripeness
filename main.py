from import_images import import_images
from MLP_classification import mlp_classification


[categories, images, test_samples] = import_images()

mlp_classification(categories, images, test_samples)
