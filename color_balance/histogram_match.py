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
import numpy as np

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

    height, width, bands = in_img.shape

    for bidx in range(bands):
        iband = in_img[:, :, bidx]
        rband = ref_img[:, :, bidx]

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
    # np.interp(in_cdf, match_cdf, range(len(in_cdf))), which
    # stretches the intensity values to min/max available intensities
    # even when matching CDF doesn't have entries at min/max intensities
    # (confirmed by unit tests)
    lut = np.arange(len(in_cdf), dtype=np.int)
    for i, c_val in enumerate(in_cdf):
        match_i = np.searchsorted(match_cdf, c_val)
        lut[i] = match_i

    # Clip to max/min values of band, as determined from cdf
    # This is necessary because np.searchsorted maps a value
    # to either 0 or len(array) if it doesn't find a target location
    max_value = np.argmax(match_cdf == match_cdf.max())
    min_value = np.argmax(match_cdf > 0)

    logging.info("clipping lut to [{},{}]".format(min_value, max_value))
    np.clip(lut, min_value, max_value, lut)

    if np.any(np.diff(lut) < 0):
        raise Exception('cdf_match lut not monotonically increasing')

    return lut.astype(np.uint8)


def _check_cdf(test_cdf):
    '''Checks that CDF monotonically increases and has a maximum value of 1'''
    if np.any(np.diff(test_cdf) < 0):
        raise CDFException('not monotonically increasing')

    if abs(test_cdf[-1] - 1.0) * 10**10 > 1:
        raise CDFException('maximum value {} not close enough to 1.0'.format(test_cdf[-1]))

    if test_cdf[0] < 0:
        raise CDFException('minimum value {} less than 0'.format(test_cdf[0]))


def mean_std_luts(in_img, ref_img, in_mask=None, ref_mask=None):
    _check_match_images(in_img, ref_img)

    height1, width1, count1 = in_img.shape
    height2, width2, count2 = ref_img.shape

    # Truthiness of color_balance masks is opposite of numpy masked arrays
    # TODO: FIX!
    in_mask = np.logical_not(in_mask.astype(bool))
    ref_mask = np.logical_not(ref_mask.astype(bool))

    in_mask = np.vstack(3 * [in_mask])
    ref_mask = np.vstack(3 * [ref_mask])

    in_tmp = in_img.reshape((height1 * width1, count1))
    ref_tmp = ref_img.reshape((height2 * width2, count2))

    in_tmp = np.ma.MaskedArray(in_tmp, mask=in_mask)
    ref_tmp = np.ma.MaskedArray(ref_tmp, mask=ref_mask)

    in_mean = in_tmp.mean(axis=0).data
    in_std = in_tmp.std(axis=0).data

    ref_mean = ref_tmp.mean(axis=0).data
    ref_std = ref_tmp.std(axis=0).data

    logging.info("Input image mean: {}" \
        .format(in_mean.tolist()))
    logging.info("Input image stddev: {}" \
        .format(in_std.tolist()))
    logging.info("Reference image mean: {}" \
        .format(ref_mean.tolist()))
    logging.info("Reference image stddev: {}" \
        .format(ref_std.tolist()))

    out_luts = []
    in_lut = np.array(range(0, 256), dtype=np.uint8)

    for bidx in range(count1):

        if in_std[bidx] == 0:
            scale = 0
        else:
            scale = float(ref_std[bidx]) / in_std[bidx]

        offset = ref_mean[bidx] - scale * in_mean[bidx]
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
