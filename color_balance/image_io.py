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

import numpy as np

from osgeo import gdal, gdal_array
from color_balance import colorimage


def load_image(image_path, band_indices=None, bit_depth=None, curve_function=None):
    """
    Loads an image into a colorimage. If no bit depth is supplied, 8-bit
    is assumed. If no band indices are supplied, RGB is assumed.
    
    .. todo:: Kill CImage
    """

    im_raster = CImage()
    im_raster.load(image_path)

    img, mask = colorimage.convert_to_colorimage(im_raster, band_indices=band_indices, bit_depth=bit_depth, curve_function=curve_function)

    return img, mask


def save_adjusted_image(filename, img, mask, cimage):
    '''Saves the colorimage to a new raster, writing the mask as the alpha
    channel and copying the geographic information from the original raster'''
    cimage.alpha = mask

    height, width, bands = img.shape

    # TODO: Why not have Cimage conform to the data structure?
    for bidx in reversed(range(bands)):
        cimage.bands[bidx] = img[:, :, bidx].astype(np.uint8)
    
    cimage.save(filename)


class CImage(object):
    """
    Geospatial image file interface.
    """


    def __init__(self):
        self.bands = []
        self.alpha = None
        self.metadata = {}

    def create_alpha(self, shape=None):
        if shape is None:
            shape = self.bands[0].shape

        self.alpha = 255 * np.ones(shape, dtype=np.uint8)

    def load(self, filename):
        gdal_ds = gdal.Open(filename)
        if gdal_ds is None:
            raise Exception('Unable to open file "{}" with gdal.Open()'.format(
                filename))

        # Read alpha band
        alpha_band = gdal_ds.GetRasterBand(gdal_ds.RasterCount)
        if alpha_band.GetColorInterpretation() == gdal.GCI_AlphaBand:
            self.alpha = alpha_band.ReadAsArray()
            band_count = gdal_ds.RasterCount - 1
        else:
            band_count = gdal_ds.RasterCount

        # 16bit TIFF files have 16bit alpha, adjust to 8bit.
        if self.alpha is not None \
                and gdal_ds.GetRasterBand(1).DataType == gdal.GDT_UInt16 \
                and np.max(self.alpha) > 255:
            self.alpha = (self.alpha / 256).astype(np.uint8)

        # Read raster bands
        for band_n in range(1, band_count+1):
            band = gdal_ds.GetRasterBand(band_n)
            array = band.ReadAsArray()
            if array is None:
                raise Exception('GDAL error occured : {}'.format(
                    gdal.GetLastErrorMsg()))
            self.bands.append(array)

        # Load georeferencing information
        self.metadata['geotransform'] = gdal_ds.GetGeoTransform()
        self.metadata['projection'] = gdal_ds.GetProjection()
        self.metadata['rpc'] = gdal_ds.GetMetadata('RPC')

    def save(self, filename):        
        band_count = len(self.bands)
        ysize, xsize = self.bands[0].shape

        options = []
        if band_count == 3:
            options.append('PHOTOMETRIC=RGB')

        if self.alpha is not None:
            band_count += 1
            options.append('ALPHA=YES')
        
        datatype = gdal_array.NumericTypeCodeToGDALTypeCode( 
            self.bands[0].dtype.type )
        
        gdal_ds = gdal.GetDriverByName('GTIFF').Create(
            filename, xsize, ysize, band_count, datatype,
            options = options)

        # Save georeferencing information
        if 'projection' in self.metadata.keys():
            gdal_ds.SetProjection(self.metadata['projection'])
        if 'geotransform' in self.metadata.keys():
            gdal_ds.SetGeoTransform(self.metadata['geotransform'])
        if 'rpc' in self.metadata.keys():
            gdal_ds.SetMetadata(self.metadata['rpc'], 'RPC')

        for i in range(len(self.bands)):
            gdal_array.BandWriteArray(
                gdal_ds.GetRasterBand(i+1),
                self.bands[i])
            
        if self.alpha is not None:
            alpha = self.alpha
            alpha_band = gdal_ds.GetRasterBand(gdal_ds.RasterCount)

            # To conform to 16 bit TIFF alpha expectations transform 
            # alpha to 16bit.
            if alpha_band.DataType == gdal.GDT_UInt16:
                alpha = ((alpha.astype(np.uint32) * 65535) / 255).astype(
                    np.uint16)
