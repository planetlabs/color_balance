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

import unittest
import numpy as np

from color_balance import colorimage, image_io as io


class Tests(unittest.TestCase):


    def test_convert_to_colorimage(self):

        # Create 4-band, 8-bit CImage. Each band has intensity values equal to
        # the band number
        test_bands = [ bidx * np.ones((5, 5), dtype=np.uint8) for bidx in range(4) ]
        test_cimage = io.CImage()
        test_cimage.bands = test_bands

        # Trying to load a 4-band image should raise an exception
        with self.assertRaises(colorimage.ImagePropertyException):
            colorimage.convert_to_colorimage(test_cimage)

        # Specifying 3 bands should allow the 4-band image to be converted
        band_indices = [0, 2, 3]
        test_colorimage, test_colormask = colorimage.convert_to_colorimage(test_cimage, band_indices)

        # The band indices are read in as RGB but the colorimage is BGR
        # so the expected image should have band order that is the reverse
        # order of band_indices.
        expected_bands = [ bidx * np.ones((5, 5), dtype=np.uint8) for bidx in reversed(band_indices) ]
        expected_image = np.dstack(expected_bands)

        np.testing.assert_array_equal(test_colorimage, expected_image)

        # CImage has no alpha channel so mask should be blank
        expected_mask = 255 * np.ones((5, 5), dtype=np.uint8)
        np.testing.assert_array_equal(test_colormask, expected_mask)

        # Change CImage alpha mask and confirm colorimage mask matches
        test_cimage.create_alpha()
        test_cimage.alpha[2, 2] = 0
        _, test_colormask = colorimage.convert_to_colorimage(test_cimage,
            band_indices=band_indices)
        expected_mask = 255 * np.ones((5, 5), dtype=np.uint8)
        expected_mask[2, 2] = 0
        np.testing.assert_array_equal(test_colormask, expected_mask)

        # Check that curve_function is applied to image
        def test_curve(r,g,b):
            return r/2, g/2, b/2
        test_colorimage, _ = colorimage.convert_to_colorimage(test_cimage,
            band_indices=band_indices, curve_function=test_curve)
        expected_bands = [i/2 * np.ones((5, 5), dtype=np.uint8) \
            for i in reversed(band_indices)]
        expected_image = np.dstack(expected_bands)
        np.testing.assert_array_equal(test_colorimage, expected_image)

        # Create 3-band, 16-bit cimage
        def np_1d_to_3d(np_1d):
            band = np.reshape(np_1d, (2, 2))
            return [band, band, band]
        test_bands = np_1d_to_3d(np.array([0, 256, 4095, 65535],
            dtype=np.uint16))
        test_cimage = io.CImage()
        test_cimage.bands=test_bands

        # No bit depth provided, should assume all 16 bits are used and divide
        # by 256 and mask should be blank
        test_colorimage, test_colormask = colorimage.convert_to_colorimage(
            test_cimage)
        expected = np_1d_to_3d(np.array([0, 1, 15, 255],
            dtype=np.uint8))
        expected_image = np.dstack(expected)
        np.testing.assert_array_equal(test_colorimage, expected_image)

        expected_mask = 255 * np.ones((2, 2), dtype=np.uint8)
        np.testing.assert_array_equal(test_colormask, expected_mask)

        # If bit depth of 12 is provided, the values should be scaled by 2**4
        # and pixels with values above 2**12 should be masked
        test_colorimage, test_colormask = colorimage.convert_to_colorimage(
            test_cimage, bit_depth=2 ** 12)
        expected = np_1d_to_3d(np.array([0, 16, 255, 255],
            dtype=np.uint8))
        expected_image = np.dstack(expected)
        np.testing.assert_array_equal(test_colorimage, expected_image)

        expected_mask = np.reshape(
            np.array([255, 255, 255, 0], dtype=np.uint8), (2, 2))
        np.testing.assert_array_equal(test_colormask, expected_mask)

        # Check that curve_function downsampling is applied to bands and avoids
        # further bit decimation
        def test_curve(r,g,b):
            r = (r/2 ** 8).astype(np.uint8)
            g = (g/2 ** 8).astype(np.uint8)
            b = (b/2 ** 8).astype(np.uint8)
            return r, g, b
        test_colorimage, _ = colorimage.convert_to_colorimage(test_cimage,
            curve_function=test_curve)
        expected_bands = np_1d_to_3d(np.array([0, 1, 15, 255],
            dtype=np.uint8))
        expected_image = np.dstack(expected_bands)
        np.testing.assert_array_equal(test_colorimage, expected_image)


    def test_get_histogram(self):

        test_band = np.array([
            [1, 1],
            [1, 1]
        ], dtype=np.uint8)
        test_mask = np.array([
            [255, 0],
            [255, 0]
        ], dtype=np.uint8)

        # All entries at intensity 1
        expected = np.zeros(256, dtype=np.int)
        expected[1] = 4
        hist = colorimage.get_histogram(test_band)
        np.testing.assert_array_equal(hist, expected)

        
        # Note: OpenCV computes an incorrect histogram array for this
        #       test case.
        expected = np.zeros(256, dtype=np.int)
        expected[1] = 2
        hist = colorimage.get_histogram(test_band, mask=test_mask)
        np.testing.assert_array_equal(hist, expected)
        
        test_band = np.array([
            [0, 1],
            [2, 3]
        ], dtype=np.uint8)
        test_mask = np.array([
            [255, 0],
            [255, 0]
        ], dtype=np.uint8)
        expected = np.zeros(256, dtype=np.int)
        expected[0] = 1
        expected[2] = 1
        
        hist = colorimage.get_histogram(test_band, mask=test_mask)
        np.testing.assert_array_equal(expected, hist)


    def test_get_cdf(self):

        test_band = np.array([
            [0, 1],
            [2, 3]
        ], dtype=np.uint8)
        test_mask = np.array([
            [255, 0],
            [255, 0]
        ], dtype=np.uint8)

        # CDF goes up by 1/4 at 0, 1, 2, and 3
        expected = np.ones(256)
        expected[0] = 0.25
        expected[1] = 0.5
        expected[2] = 0.75

        cdf = colorimage.get_cdf(test_band)
        np.testing.assert_array_equal(cdf, expected)

        # 1 and 3 masked, so CDF goes up 1/2 at 0 and 2
        expected = np.ones(256)
        expected[0] = 0.5
        expected[1] = 0.5

        cdf = colorimage.get_cdf(test_band, mask=test_mask)
        np.testing.assert_array_equal(expected, cdf)


    def test_scale_offset_lut(self):
        test_lut = np.array(range(10))

        # Test that the returned lut equals the entered lut if
        # no scale or offset is given
        expected = test_lut
        lut = colorimage.scale_offset_lut(test_lut)
        np.testing.assert_array_equal(lut, expected)

        # Test clipping at top values
        expected = np.array([5, 6, 7, 8, 9, 9, 9, 9, 9, 9])
        lut = colorimage.scale_offset_lut(test_lut, offset=5)
        np.testing.assert_array_equal(lut, expected)

        # Test clipping at bottom values
        expected = np.array([0, 0, 0, 0, 0, 0, 1, 2, 3, 4])
        lut = colorimage.scale_offset_lut(test_lut, offset=-5)
        np.testing.assert_array_equal(lut, expected)

        # Test scaling by less than one
        expected = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        lut = colorimage.scale_offset_lut(test_lut, scale=0.5)
        np.testing.assert_array_equal(lut, expected)

        # Test scaling by greater than one
        expected = np.array([0, 2, 4, 6, 8, 9, 9, 9, 9, 9])
        lut = colorimage.scale_offset_lut(test_lut, scale=2)
        np.testing.assert_array_equal(lut, expected)

        # Test for clipping effects
        # Results should be the same as test 1 above
        test_lut = 2 * np.array(range(10))
        expected = np.array([5, 6, 7, 8, 9, 9, 9, 9, 9, 9])
        lut = colorimage.scale_offset_lut(test_lut, scale=0.5, offset=5)
        np.testing.assert_array_equal(lut, expected)


    def test_apply_lut(self):
        test_band = np.array([[0, 1], [2, 3]], dtype=np.uint8)

        test_lut = np.array(range(256), dtype=np.uint8)
        test_lut[test_lut < 256 - 5] += 5
        expected = np.array([[5, 6], [7, 8]], dtype=np.uint8)
        new_band = colorimage.apply_lut(test_band, test_lut)
        np.testing.assert_array_equal(new_band, expected)

        test_lut = np.array(range(256), dtype=np.uint8)
        valid = test_lut >= 5
        test_lut[valid] -= 5
        test_lut[~valid] = 0
        expected = np.array([[0, 0], [0, 0]], dtype=np.uint8)
        new_band = colorimage.apply_lut(test_band, test_lut)
        np.testing.assert_array_equal(new_band, expected)


if __name__ == '__main__':
    unittest.main()
