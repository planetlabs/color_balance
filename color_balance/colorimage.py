import numpy
import cv2

from osgeo import gdal, gdal_array


class OutOfRangeException(Exception):
    pass


class ImagePropertyException(Exception):
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
        mask = cimage.alpha.astype(numpy.uint8)
    else:
        mask = create_mask(cimage.bands[0])

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
        mask[clip_max] = 0
        clip_min = band < 0
        mask[clip_min] = 0
        out_bands.append(band.astype(numpy.uint8))

    # Convert to BGR to images are BGR.
    return cv2.merge(list(reversed(out_bands))), mask


def create_mask(band, value=None):
    '''Creates an openCV mask of the same size as the input band. If a value
    is provided, the locations of pixels equal to the value are added to the
    mask.'''
    mask = 255 * numpy.ones(band.shape, dtype=numpy.uint8)
    if value is not None:
        cond = band == value
        mask[cond] = 0
    return mask


def combine_masks(masks):
    '''Combines masks into one mask'''
    mask = masks[0]
    for add_mask in masks[1:]:
        mask[add_mask == 0] = 0
    return mask


def map_masked(img, mask, value=0):
    '''Maps intensity values of pixels that are masked to a new value'''
    channels = []
    for channel in cv2.split(img):
        channel[mask == 0] = value
        channels.append(channel)
    img = cv2.merge(channels)
    return img


def get_mask_percent(mask):
    '''Helper function for determining how many pixels are masked.'''
    cond = mask == 0
    num_masked = numpy.extract(cond, mask).size
    mask_percent = float(100 * num_masked) / mask.size
    return mask_percent


def get_histogram(band, mask=None, normalized=False):
    '''Calculate the histogram of a band. If a mask is provided, masked pixels
    are not considered in the histogram calculation. If normalized is True, the
    histogram entries are the count divided by the total number of pixels,
    turning it into a distribution function.

    :param numpy array band: source for calculating histogram
    :param numpy array mask: array containing zero values at locations where
        pixels should not be considered, 255 elsewhere
    :param boolean normalized: determines whether histogram pixel count will be
        returned (if False) or relative frequency (if True)
    :returns: count (or freqency) of pixels at each possible intensity value
    '''
    if band.max() > 255 or band.min() < 0:
        raise OutOfRangeException("Band values outside of [0, 255]")

    bit_depth = 256
    hist = cv2.calcHist([band], [0], mask, [bit_depth], [0, bit_depth])
    if normalized:
        # Turn histogram to pdf by normalizing to sum to 1
        hist = hist * (1.0 / hist.sum())
    return hist


def get_cdf(band, mask=None):
    '''Calculate the cumulative distribution function for the band. If a mask
    is provided, masked pixels are not considered in the calculation.

    :param numpy array band: source for calculating cdf
    :param numpy array mask: array containing zero values at locations where
        pixels should not be considered, 255 elsewhere
    :returns: cumulative sum of pixels at each possible intensity value
    '''
    hist = get_histogram(band, mask, normalized=True)
    cdf = hist.cumsum()
    return cdf


def cdf_match_lut(in_cdf, match_cdf):
    '''Create a look up table for matching the input cdf to the match cdf. At
    each intensity, this algorithm gets the value of the input cdf, then finds
    the intensity at which the match cdf has the same value.'''

    if numpy.any(numpy.diff(in_cdf) < 0) or \
       numpy.any(numpy.diff(match_cdf < 0)):
        raise Exception("Not a valid cdf - should monotonically increase.")
    assert len(in_cdf) == len(match_cdf), \
        "cdfs don't have same number of entries"

    # This approach is preferred over using
    # numpy.interp(in_cdf, match_cdf, range(len(in_cdf))), which
    # stretches the intensity values to min/max available intensities
    # even when matching CDF doesn't have entries at min/max intensities
    # (confirmed by unit tests)
    lut = numpy.zeros(len(in_cdf), dtype=numpy.uint8)
    for i, c_val in enumerate(in_cdf):
        match_i = numpy.searchsorted(match_cdf, c_val)
        lut[i] = match_i

    return lut


def scale_offset_lut(in_lut, scale=1.0, offset=0):
    '''Creates a lut that maps intensity to new values based on scale and
    offset but clips the new values at the intensity extents.'''
    out_lut = in_lut.copy()

    # No adjustments are necessary, return lut copy
    if scale == 1 and offset == 0:
        return out_lut

    min_val = 0
    max_val = len(out_lut) - 1
    for i, intensity in enumerate(in_lut):
        new_int = (1.0 * scale) * intensity + offset

        # Clip lut values to min/max
        out_lut[i] = min(max(new_int, min_val), max_val)
    return out_lut


def apply_lut(band, lut):
    '''Changes band intensity values based on intensity look up table (lut)'''
    assert lut.dtype == band.dtype, ("Band ({}) and lut ({}) must be the " +
                                     "same data type."
                                     ).format(band.dtype, lut.dtype)
    return numpy.take(lut, band, mode='clip')
