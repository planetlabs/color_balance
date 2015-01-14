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
        lut = ci.cdf_match_lut(in_cdf, ref_cdf)
        out_luts.append(lut)
    return out_luts


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
