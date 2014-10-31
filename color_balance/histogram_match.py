import math

import numpy
import cv2

from color_balance import colorimage as ci


def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)


HIST_MATCH_METHOD = enum('CDF_NORM', 'MEAN_STD')
HIST_DISTANCE = enum('CHI_SQUARED', 'CORRELATION', 'JENSEN_SHANNON')


def match_histogram(in_img, ref_img, in_mask=None, ref_mask=None,
                    hist_match_method=HIST_MATCH_METHOD.CDF_NORM):
    '''Matches the histograms of the bands of the input image to the histograms
    of the bands of the reference image.'''
    # Assumptions
    max_val = 255
    min_val = 0
    assert in_img.shape[2] == ref_img.shape[2]
    for image in [in_img, ref_img]:
        if image.max() > max_val or image.min() < min_val:
            raise ci.OutOfRangeException("Image values outside of [0, 255]")

    def cdf_norm(in_img, in_mask, ref_img, ref_mask):
        out_bands = []
        for iband, rband in zip(cv2.split(in_img), cv2.split(ref_img)):
            in_cdf = ci.get_cdf(iband, mask=in_mask)
            ref_cdf = ci.get_cdf(rband, mask=ref_mask)
            lut = ci.cdf_match_lut(in_cdf, ref_cdf)
            out_bands.append(ci.apply_lut(iband, lut))
        return cv2.merge(out_bands)

    def mean_std(in_img, in_mask, ref_img, ref_mask):
        in_mean, in_std = cv2.meanStdDev(in_img, mask=in_mask)
        ref_mean, ref_std = cv2.meanStdDev(ref_img, mask=ref_mask)

        out_bands = []
        in_lut = numpy.array(range(min_val, max_val+1), dtype=numpy.uint8)
        for i, band in enumerate(cv2.split(in_img)):
            scale = float(ref_std[i])/in_std[i]
            offset = ref_mean[i] - scale*in_mean[i]
            lut = ci.scale_offset_lut(in_lut, scale=scale, offset=offset)
            out_bands.append(ci.apply_lut(band, lut))

        return cv2.merge(out_bands)

    fcns = {HIST_MATCH_METHOD.CDF_NORM: cdf_norm,
            HIST_MATCH_METHOD.MEAN_STD: mean_std}
    try:
        fcn = fcns[hist_match_method]
    except KeyError:
        raise Exception("Unexpected hist_match_method")

    return fcn(in_img, in_mask, ref_img, ref_mask)


def histogram_distances(image1, image2, distance=HIST_DISTANCE.CHI_SQUARED):
    '''Measures the distance between the histogram of corresponding bands in
    two images. '''
    assert len(image1.bands) == len(image2.bands)
    dist = []
    for i in range(len(image1.bands)):
        dist.append(histogram_distance(image1.bands[i], image2.bands[i],
                                       distance))
    return dist


def histogram_distance(band1, band2, distance=HIST_DISTANCE.CHI_SQUARED):
    '''Measures the distance between the histogram of two bands.'''
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

    fcns = {HIST_DISTANCE.CHI_SQUARED: chi_squared,
            HIST_DISTANCE.CORRELATION: correlation,
            HIST_DISTANCE.JENSEN_SHANNON: jensen_shannon}

    assert distance in fcns.keys()

    histA = band1.get_histogram()
    histB = band2.get_histogram()

    assert histA.shape == histB.shape

    return fcns[distance](histA, histB)
