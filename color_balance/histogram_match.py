"""
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
"""

import logging
import numpy as np

from color_balance import colorimage as ci


class CDFException(Exception):
    pass


def match_histogram(luts_calculation_function, in_img, ref_img,
        in_mask=None, ref_mask=None):
    '''Runs luts_calculation_function on in_img and ref_img (using in_mask and
    ref_mask, if provided) and applies luts to in_img.'''
    _check_match_images(in_img, ref_img, in_img.dtype)
    luts = luts_calculation_function(in_img, ref_img, in_mask, ref_mask)
    matched_img = ci.apply_luts(in_img, luts)
    return matched_img


def cdf_normalization_luts(in_img, ref_img, in_mask=None, ref_mask=None, dtype=np.uint8):

    out_luts = []
    height, width, bands = in_img.shape

    for bidx in range(bands):
        iband = in_img[:, :, bidx]
        rband = ref_img[:, :, bidx]

        in_cdf = ci.get_cdf(iband, mask=in_mask)
        ref_cdf = ci.get_cdf(rband, mask=ref_mask)
        #print bidx, in_cdf.sum(), ref_cdf.sum()
        lut = cdf_match_lut(in_cdf, ref_cdf, dtype=dtype)
        out_luts.append(lut)

    return out_luts

def _spread_cdf(cdf, target_size):
    assert len(cdf) == 255
    assert target_size == 4095

    # Spoof a 12 bit domain over an 8 bit number of points
    x = np.linspace(0, 4095, 255)

    # Create the 12 bit domain
    xp = np.linspace(0, 4095, target_size)

    # Interpolate the points in between
    return np.interp(xp, x, cdf)


def cdf_match_lut(src_cdf, ref_cdf, dtype=np.uint8):
    """
    Create a look up table for matching the source CDF to the reference CDF. At
    each intensity, this algorithm gets the value of the source CDF, then finds
    the intensity at which the reference CDF has the same value.

    :param src_cdf:
        Source CDF represented as a 1d array.
    :param ref_cdf:
        Reference CDF represented as a 1d array.
    :param dtype:
        Numpy data type used to represent the output LUT
    :return look up table:
    """

    _check_cdf(src_cdf)
    _check_cdf(ref_cdf)

    # If we have an 8 bit target image we need to spread out the CDF
    # artificially to use relative to 12 bit values.
    if len(src_cdf) == 4095 and len(ref_cdf) == 255:
        ref_cdf = _spread_cdf(ref_cdf, len(src_cdf))

    # TODO: Why enforce the lengths to be the same? Needed for current approach,
    #       but limiting algo to fixed-bin histograms.
    assert len(src_cdf) == len(ref_cdf), \
        "CDFs don't have same number of entries"

    # This approach is preferred over using
    # np.interp(src_cdf, ref_cdf, range(len(src_cdf))), which
    # stretches the intensity values to min/max available intensities
    # even when matching CDF doesn't have entries at min/max intensities
    # (confirmed by unit tests)

    lut = np.searchsorted(ref_cdf, src_cdf).astype(dtype)

    # Clip to max/min values of band, as determined from CDF
    # This is necessary because np.searchsorted maps a value
    # to either 0 or len(array) if it doesn't find a target location
    max_value = np.argmax(ref_cdf == ref_cdf.max())
    min_value = np.argmax(ref_cdf > 0)

    logging.info("clipping lut to [{},{}]".format(min_value, max_value))
    np.clip(lut, min_value, max_value, lut)

    if np.any(np.diff(lut) < 0):
        raise Exception('ref_cdf lut not monotonically increasing')

    return lut


def _check_cdf(cdf):
    """
    Checks that CDF monotonically increases and has a maximum value of 1.

    .. todo:: Is this check necessary? Will np.cumsum() ever return a non-monotonically increasing function?
    """

    if np.any(np.diff(cdf) < 0):
        raise CDFException('not monotonically increasing')

    if abs(cdf[-1] - 1.0) * 10**10 > 1:
        raise CDFException('maximum value {} not close enough to 1.0'.format(cdf[-1]))

    if cdf[0] < 0:
        raise CDFException('minimum value {} less than 0'.format(cdf[0]))


def mean_std_luts(in_img, ref_img, in_mask=None, ref_mask=None, dtype=np.uint16):

    if in_mask is not None:
        in_mask = np.tile(in_mask[..., None], (1, 1, in_img.shape[2]))
        in_img = np.ma.masked_where(in_mask, in_img)

    if ref_mask is not None:
        ref_mask = np.tile(ref_mask[..., None], (1, 1, ref_img.shape[2]))
        ref_img = np.ma.masked_where(ref_mask, ref_img)

    # Need to make sure we check after masking invalid values
    # Some color targets have values out of the 12-bit range.
    _check_match_images(in_img, ref_img, dtype)

    nbands = in_img.shape[2]
    in_mean = np.array([in_img[..., i].mean() for i in range(nbands)])
    in_std = np.array([in_img[..., i].std() for i in range(nbands)])
    ref_mean = np.array([ref_img[..., i].mean() for i in range(nbands)])
    ref_std = np.array([ref_img[..., i].std() for i in range(nbands)])

    logging.info("Input image mean: {}" \
        .format(in_mean.tolist()))
    logging.info("Input image stddev: {}" \
        .format(in_std.tolist()))
    logging.info("Reference image mean: {}" \
        .format(ref_mean.tolist()))
    logging.info("Reference image stddev: {}" \
        .format(ref_std.tolist()))

    out_luts = []

    if dtype == np.uint16:
        # Assume any 16-bit images are actually 12-bit.
        minimum = 0
        maximum = 4096
    else:
        minimum = np.iinfo(dtype).min
        maximum = np.iinfo(dtype).max
    in_lut = np.arange(minimum, maximum + 1, dtype=dtype)

    # Need to rescale for 8-bit color targets...
    if ref_img.dtype != dtype:
        dmin = np.iinfo(ref_img.dtype).min
        dmax = np.iinfo(ref_img.dtype).max

        ref_mean = maximum * (ref_mean - dmin) / float(dmax - dmin)
        ref_std = maximum * ref_std / float(dmax - dmin)

    for bidx in range(3):

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


def _check_match_images(in_img, ref_img, dtype):
    if in_img.shape[2] != ref_img.shape[2]:
        raise MatchImagesException("Images must have the same number of bands")

    minimum = 0
    maximum = 255 if dtype == np.uint8 else 4095

    for image in [in_img, ref_img]:
        if image.max() > maximum or image.min() < minimum:
            raise MatchImagesException("Image values outside of [%d, %d]" % (minimum, maximum))
