
import numpy as np
from color_balance import colorimage as ci


def image_distances(distance_function, image1, image2):
    """
    Measures the distance between the histogram of corresponding bands in
    two images.
    """
    
    shape1 = image1.shape
    shape2 = image2.shape
    
    assert shape1[2] == shape2[2]
    
    bands1 = [ image1[:, :, bidx] for bidx in range(shape1[2]) ]
    bands2 = [ image2[:, :, bidx] for bidx in range(shape2[2]) ]

    dist = []
    for band1, band2 in zip(bands1, bands2):
        dist.append(distance_function(
            ci.get_histogram(band1), ci.get_histogram(band2)))
    return dist


def chi_squared(hist1, hist2):
    return np.sum((hist1 - hist2) ** 2 / hist1)


def correlation(hist1, hist2):

    avg1 = np.mean(hist1.astype(np.float32))
    avg2 = np.mean(hist2.astype(np.float32))

    d1 = hist1 - avg1
    d2 = hist2 - avg2

    numerator = np.sum(d1 * d2)
    denominator = np.sqrt(np.sum(d1 ** 2) * np.sum(d2 ** 2))

    return numerator / denominator


def jensen_shannon(hist1, hist2):

    def entropy(prob_dist):
        return -sum([p * np.log(p, 2) for p in prob_dist if p != 0])

    js_left = hist1 + hist2
    js_right = entropy(hist1) + entropy(hist2)

    return entropy(js_left) - js_right
