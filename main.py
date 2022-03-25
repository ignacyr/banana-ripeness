from import_images import import_images
from MLP_classification import mlp_classification
from SVC_classification import svc_classification
from decision_tree_classification import decision_tree_classification


resolution = 50  # don't set to more than 100
[categories, images, test_samples] = import_images(resolution)

decision_tree_classification(categories, images, test_samples, resolution)
svc_classification(categories, images, test_samples, resolution)
mlp_classification(categories, images, test_samples, resolution)
