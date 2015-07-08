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

import numpy as np
from osgeo import gdal


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

    mask = 255 * np.ones_like(band, dtype=np.uint8)

    if value:
        mask[band == value] = 0

    return mask


def map_masked(img, mask, value=0):
    """
    Maps intensity values of pixels that are masked to a new value
    
    .. todo:: operation might be doable with array broadcasting.
    """
    
    shape = img.shape
    img = img.reshape((shape[0], shape[1], -1))

    height, width, bands = img.shape
    
    for bidx in range(bands):
        img[:, :, bidx][mask == 0] = value
    
    return img.reshape(shape)


def get_mask_percent(mask):
    '''Helper function for determining how many pixels are masked.'''
    cond = mask == 0
    num_masked = np.extract(cond, mask).size
    mask_percent = float(100 * num_masked) / mask.size
    return mask_percent
