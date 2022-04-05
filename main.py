from import_images import import_images
from MLP_classification import mlp_classification
from SVC_classification import svc_classification
from decision_tree_classification import decision_tree_classification
from numpy import load


def main():
    resolution = 100  # don't set to more than 100
    # import_images(resolution)

    categories = load('categories.npy')
    images = load('images.npy')
    test_samples = load('test_samples_img.npy')

    decision_tree_classification(categories, images, test_samples, resolution)
    svc_classification(categories, images, test_samples, resolution)
    mlp_classification(categories, images, test_samples, resolution)


if __name__ == '__main__':
    main()
