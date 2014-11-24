import math

import numpy
import cv2

from color_balance import colorimage as ci

def image_distances(distance_function, image1, image2):
    '''Measures the distance between the histogram of corresponding bands in
    two images. '''
    bands1 = cv2.split(image1)
    bands2 = cv2.split(image2)
    assert len(bands1) == len(bands2)
    dist = []
    for band1, band2 in zip(bands1, bands2):
        dist.append(distance_function(
            ci.get_histogram(band1), ci.get_histogram(band2)))
    return dist


def chi_squared(histA, histB):
    return cv2.compareHist(histA.astype(numpy.float32),
                           histB.astype(numpy.float32),
                           cv2.cv.CV_COMP_CHISQR)


def correlation(histA, histB):
    return cv2.compareHist(histA.astype(numpy.float32),
                           histB.astype(numpy.float32),
                           cv2.cv.CV_COMP_CORREL)


def jensen_shannon(histA, histB):
    def entropy(prob_dist):
        return -sum([p * math.log(p, 2) for p in prob_dist if p != 0])

    js_left = histA + histB
    js_right = entropy(histA) + entropy(histB)
    return entropy(js_left) - js_right
