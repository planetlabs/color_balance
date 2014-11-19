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
import unittest
import os

import numpy

from osgeo import gdal, gdal_array

from color_balance import image_io as io


class CImageTests(unittest.TestCase):
    def _create_image_with_geotransform(self, filename):
        gdal_ds = gdal.GetDriverByName('GTiff').Create(
            filename, 2, 2, 1, gdal.GDT_UInt16)
        gdal_array.BandWriteArray(
            gdal_ds.GetRasterBand(1),
            numpy.ones((2,2), dtype=numpy.uint16))
        gdal_ds.SetGeoTransform([-1.0, 1.0, 0.0, 1.0, 0.0, -1.0])

    def test_create_alpha(self):
        shape = (5, 5)
        expected_alpha = 255 * numpy.ones((5, 5), dtype=numpy.uint8)

        alpha_from_shape_cimage = io.CImage()
        alpha_from_shape_cimage.create_alpha(shape)
        numpy.testing.assert_array_equal(
            alpha_from_shape_cimage.alpha, expected_alpha)

        band = numpy.ones((5, 5), dtype=numpy.uint8)
        alpha_from_band_cimage = io.CImage()
        alpha_from_band_cimage.bands=[band]
        alpha_from_band_cimage.create_alpha()
        numpy.testing.assert_array_equal(
            alpha_from_band_cimage.alpha, expected_alpha)

    def test_load(self):
        input_file = 'test_file.tif'
        self._create_image_with_geotransform(input_file)
        test_cimage = io.CImage()
        test_cimage.load(input_file)

        test_geotransform = test_cimage.metadata['geotransform']
        self.assertEquals(test_geotransform, (-1.0, 1.0, 0.0, 1.0, 0.0, -1.0))

        os.unlink(input_file)

    def test_save(self):
        output_file = 'test_file.tif'
        test_band = 255 * numpy.ones((5, 5), dtype=numpy.uint8)
        test_cimage = io.CImage()
        test_cimage.bands = [test_band]
        test_cimage.save(output_file)

        test_ds = gdal.Open(output_file)

        saved_number_of_bands = test_ds.RasterCount
        self.assertEquals(saved_number_of_bands, 1)

        saved_band = test_ds.GetRasterBand(1).ReadAsArray()
        numpy.testing.assert_array_equal(saved_band, test_band)
        
        os.unlink(output_file)
 

if __name__ == '__main__':
    unittest.main()
