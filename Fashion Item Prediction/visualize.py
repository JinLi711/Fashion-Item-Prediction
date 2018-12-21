import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import confusion_matrix

def plot_items(instances, images_per_row=10, **options):
    """
    Plot the items from the instances. The image must be of size 28*28.

    :param instances: A 2d array of (instances, 28*28).
    :type  instances: numpy.ndarray.
    :param images_per_row: Number of images per row
    :type  images_per_row: int
    """

    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))

    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))

    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")

def conf_mx_rates (y, y_pred):
    """
    Given labels and predictions, creates a confusion matrix of error rates.
    Each row is an actual class, while each column is a predicted class.
    The whiter the square, the more the image is misclassified

    :param y: The labels
    :type  y: pandas.core.series.Series
    :param y_pred: The predictions based on the ML algorithm.
    :type  y_pred: pandas.core.series.Series
    """
    conf_mx = confusion_matrix (y, y_pred)
    row_sums = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx / row_sums
    
    np.fill_diagonal(norm_conf_mx, 0)
    plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
    plt.show()