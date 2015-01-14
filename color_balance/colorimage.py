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
import warnings

import numpy
import cv2

from osgeo import gdal, gdal_array

from color_balance import mask


class OutOfRangeException(Exception):
    pass


class ImagePropertyException(Exception):
    pass


class LUTException(Exception):
    pass


def convert_to_colorimage(cimage, band_indices=None,
                          bit_depth=None, curve_function=None):
    '''Creates a 3-band, 8-bit BGR colorimage from the bands of the input RGB
    CImage. If band indices are provided, they are used to identify the red,
    green, and blue bands of the CImage, respectively. If the CImage is
    16-bit, the intensity values of the bands are scaled to 8-bit. In this
    case, the bit depth is assumed to be 16 unless bit depth is provided.
    If the CImage has an alpha channel, it is converted to the colorimage
    mask.

    :param CImage cimage: CImage to be converted
    :param list band_indices: ordered list of indices for red, green, and blue
        channels
    :param int bit_depth: effective bit depth, if less than actual CImage bit
        depth
    :param function curve_function: function for applying a color curve to the
        RGB bands
    :returns: colorimage (3-band, 8-bit BRG openCV array), mask (1-band, 8-bit
        array where 0 corresponds to masked pixels and 255 corresponds to
        unmasked pixels), image (same as input CImage)
    '''
    if band_indices is None:
        if len(cimage.bands) != 3:
            raise ImagePropertyException("CImage is not 3-band")
        band_indices = [0, 1, 2]
    else:
        assert len(band_indices) == 3
        assert max(band_indices) < len(cimage.bands)
        assert min(band_indices) >= 0

    # Make a mask from the CImage alpha channel
    if cimage.alpha is not None:
        cmask = cimage.alpha.astype(numpy.uint8)
    else:
        cmask = mask.create_mask(cimage.bands[0])

    bands = []
    for i in band_indices:
        bits16 = gdal_array.NumericTypeCodeToGDALTypeCode(
            cimage.bands[i].dtype.type) != gdal.GDT_Byte

        # Make RGB band image
        ntype = numpy.uint16 if bits16 else numpy.uint8
        cimage.bands[i].astype(ntype).dtype
        bands.append(cimage.bands[i].astype(ntype))

    # Apply curve function to RGB image
    if curve_function is not None:
        bands = curve_function(*bands)

    # Scale the band intensities and mask out-of-range pixels
    out_bands = []
    for band in bands:
        # If 16-bit, convert to 8-bit, assuming 16-bit image is using the
        # entire bit depth unless bit depth is given
        if band.dtype == numpy.uint16:
            if bit_depth is not None:
                band = (band / (bit_depth / 2 ** 8))
            else:
                band = (band / 2 ** 8)
        elif band.dtype != numpy.uint8:
            raise ImagePropertyException(
                "Band type is {}, should be {} or {}".
                format(band.dtype,
                       numpy.uint8, numpy.uint16))

        # Mask out values above or below the 8-bit extents
        # This is only necessary if bit depth is given
        clip_max = band > 255
        cmask[clip_max] = 0
        clip_min = band < 0
        cmask[clip_min] = 0
        out_bands.append(band.astype(numpy.uint8))

    cimg = rgb_bands_to_colorimage(out_bands)
    return cimg, cmask


def rgb_bands_to_colorimage(rgb_bands):
    return cv2.merge(list(reversed(rgb_bands)))


def colorimage_to_rgb_bands(c_img):
    return reversed(cv2.split(c_img))


def get_histogram(band, mask=None):
    '''Calculate the histogram of a band. If a mask is provided, masked pixels
    are not considered in the histogram calculation.

    :param numpy array band: source for calculating histogram
    :param numpy array mask: array containing zero values at locations where
        pixels should not be considered, 255 elsewhere
    :returns: count of pixels at each possible intensity value
    '''
    if band.max() > 255 or band.min() < 0:
        raise OutOfRangeException("Band values outside of [0, 255]")

    bit_depth = 256
    hist = cv2.calcHist([band], [0], mask, [bit_depth], [0, bit_depth])

    # cv2.calcHist returns a 2D float array
    # Convert histogram to 1D array of ints
    int_hist = hist[:,0].astype(numpy.int)
    return int_hist


def get_cdf(band, mask=None):
    '''Calculate the cumulative distribution function for the band. If a mask
    is provided, masked pixels are not considered in the calculation.

    :param numpy array band: source for calculating cdf
    :param numpy array mask: array containing zero values at locations where
        pixels should not be considered, 255 elsewhere
    :returns: cumulative sum of pixels at each possible intensity value
    '''
    hist = get_histogram(band, mask)

    # If the datatype of the histogram is not int, CDF calculation is off, for
    # example, CDF may never reach 1
    assert hist.dtype == numpy.int

    normalized_hist = hist * (1.0 / hist.sum())
    cdf = normalized_hist.cumsum()
    return cdf


def _check_cdf(test_cdf):
    '''Checks that CDF monotonically increases and has a maximum value of 1'''
    warnings.warn("deprecated - moved to histogram_match", DeprecationWarning)


def cdf_match_lut(in_cdf, match_cdf):
    '''Create a look up table for matching the input cdf to the match cdf. At
    each intensity, this algorithm gets the value of the input cdf, then finds
    the intensity at which the match cdf has the same value.'''
    warnings.warn("deprecated - moved to histogram_match", DeprecationWarning)


def scale_offset_lut(in_lut, scale=1.0, offset=0):
    '''Creates a lut that maps intensity to new values based on scale and
    offset but clips the new values at the intensity extents.'''
    # No adjustments are necessary, return lut copy
    if scale == 1 and offset == 0:
        logging.info("skipped adjusting lut because scale is 1 and offset is 0")
        return in_lut.copy()

    logging.info("adjusting lut with scale {} and offset {}" \
        .format(float(scale), float(offset)))
    out_lut = (1.0 * scale) * in_lut + offset

    min_val = 0
    max_val = len(out_lut) - 1
    logging.info("clipping lut to [{},{}]".format(min_val, max_val))
    numpy.clip(out_lut, min_val, max_val, out_lut)
    return out_lut.astype(numpy.uint8)


def apply_lut(band, lut):
    '''Changes band intensity values based on intensity look up table (lut)'''
    if lut.dtype != band.dtype:
        raise LUTException("Band ({}) and lut ({}) must be the same data " +
            "type.").format(band.dtype, lut.dtype)
    return numpy.take(lut, band, mode='clip')


def apply_luts(image, luts):
    '''Changes intensity values for each band in input image based on intensity
    look up tables (luts)'''
    in_bands = cv2.split(image)
    if len(in_bands) != len(luts):
        raise LUTException("image bands ({}) and lut ({}) must have the same" +
            " number of entries.".format(len(image), len(luts)))

    out_bands = [apply_lut(band, lut) \
        for (band, lut) in zip(in_bands, luts)]
    return cv2.merge(out_bands)
