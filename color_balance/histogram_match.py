'''
Copyright 2014 Planet Labs, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
import logging

import numpy
import cv2

from color_balance import colorimage as ci


class CDFException(Exception):
    pass


def match_histogram(luts_calculation_function, in_img, ref_img,
        in_mask=None, ref_mask=None):
    '''Runs luts_calculation_function on in_img and ref_img (using in_mask and
    ref_mask, if provided) and applies luts to in_img.'''
    _check_match_images(in_img, ref_img)
    luts = luts_calculation_function(in_img, ref_img, in_mask, ref_mask)
    matched_img = ci.apply_luts(in_img, luts)
    return matched_img


def cdf_normalization_luts(in_img, ref_img, in_mask=None, ref_mask=None):
    out_luts = []
    for iband, rband in zip(cv2.split(in_img), cv2.split(ref_img)):
        in_cdf = ci.get_cdf(iband, mask=in_mask)
        ref_cdf = ci.get_cdf(rband, mask=ref_mask)
        lut = cdf_match_lut(in_cdf, ref_cdf)
        out_luts.append(lut)
    return out_luts


def cdf_match_lut(in_cdf, match_cdf):
    '''Create a look up table for matching the input cdf to the match cdf. At
    each intensity, this algorithm gets the value of the input cdf, then finds
    the intensity at which the match cdf has the same value.'''

    _check_cdf(in_cdf)
    _check_cdf(match_cdf)
    assert len(in_cdf) == len(match_cdf), \
        "cdfs don't have same number of entries"

    # This approach is preferred over using
    # numpy.interp(in_cdf, match_cdf, range(len(in_cdf))), which
    # stretches the intensity values to min/max available intensities
    # even when matching CDF doesn't have entries at min/max intensities
    # (confirmed by unit tests)
    lut = numpy.arange(len(in_cdf), dtype=numpy.int)
    for i, c_val in enumerate(in_cdf):
        match_i = numpy.searchsorted(match_cdf, c_val)
        lut[i] = match_i

    # Clip to max/min values of band, as determined from cdf
    # This is necessary because numpy.searchsorted maps a value
    # to either 0 or len(array) if it doesn't find a target location
    max_value = numpy.argmax(match_cdf == match_cdf.max())
    min_value = numpy.argmax(match_cdf > 0)

    logging.info("clipping lut to [{},{}]".format(min_value, max_value))
    numpy.clip(lut, min_value, max_value, lut)

    if numpy.any(numpy.diff(lut) < 0):
        raise Exception('cdf_match lut not monotonically increasing')

    return lut.astype(numpy.uint8)


def _check_cdf(test_cdf):
    '''Checks that CDF monotonically increases and has a maximum value of 1'''
    if numpy.any(numpy.diff(test_cdf) < 0):
        raise CDFException('not monotonically increasing')

    if abs(test_cdf[-1] - 1.0) * 10**10 > 1:
        raise CDFException('maximum value {} not close enough to 1.0'.format(test_cdf[-1]))

    if test_cdf[0] < 0:
        raise CDFException('minimum value {} less than 0'.format(test_cdf[0]))


def mean_std_luts(in_img, ref_img, in_mask=None, ref_mask=None):
    _check_match_images(in_img, ref_img)

    in_mean, in_std = cv2.meanStdDev(in_img, mask=in_mask)
    ref_mean, ref_std = cv2.meanStdDev(ref_img, mask=ref_mask)
    logging.info("Input image mean: {}" \
        .format([float(m) for m in in_mean]))
    logging.info("Input image stddev: {}" \
        .format([float(s) for s in in_std]))
    logging.info("Reference image mean: {}" \
        .format([float(m) for m in ref_mean]))
    logging.info("Reference image stddev: {}" \
        .format([float(s) for s in ref_std]))

    out_luts = []
    in_lut = numpy.array(range(0, 256), dtype=numpy.uint8)
    for i, band in enumerate(cv2.split(in_img)):
        if in_std[i] == 0:
            scale = 0
        else:
            scale = float(ref_std[i])/in_std[i]
        offset = ref_mean[i] - scale*in_mean[i] 
        lut = ci.scale_offset_lut(in_lut, scale=scale, offset=offset)
        out_luts.append(lut)
    return out_luts


class MatchImagesException(Exception):
    pass


def _check_match_images(in_img, ref_img):
    if in_img.shape[2] != ref_img.shape[2]:
        raise MatchImagesException("Images must have the same number of bands")
   
    max_val = 255
    min_val = 0
    for image in [in_img, ref_img]:
        if image.max() > max_val or image.min() < min_val:
            raise MatchImagesException("Image values outside of [0, 255]")
