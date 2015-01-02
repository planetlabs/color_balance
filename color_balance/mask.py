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
import numpy
from osgeo import  gdal
import cv2


def load_mask(mask_path, conservative=True):
    '''Loads a single-band mask image and adds it to the mask array.
    If conservative is True, only totally-masked pixels are masked out.
    Otherwise, all partially pixels are masked out.'''
    gdal_ds = gdal.Open(mask_path)
    if gdal_ds is None:
        raise IOError('Mask could not be opened.')

    band = gdal_ds.GetRasterBand(1)
    mask_array = band.ReadAsArray()

    if conservative:
        # Partially-masked pixels set to unmasked
        mask_array[mask_array > 0] = 255
    else:
        # Partially-masked pixels set to masked
        mask_array[mask_array < 255] = 0

    return mask_array


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
