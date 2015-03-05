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
import unittest

import numpy
import cv2

from color_balance import histogram_match as hm


class Tests(unittest.TestCase):
    def setUp(self):
        sequence_band = numpy.array(range(0,256), dtype=numpy.uint8)
        first_half_band = numpy.array(range(0,128)*2, dtype=numpy.uint8)
        second_half_band = numpy.array(range(128,256)*2, dtype=numpy.uint8)
        spread_band = numpy.array(range(0,256,32)*32, dtype=numpy.uint8)
        self.constant_band = numpy.ones(256, dtype=numpy.uint8)

        self.compressed_img = cv2.merge(
            [spread_band, first_half_band, second_half_band])

        self.sequence_img = cv2.merge([sequence_band]*3)
        self.constant_img = cv2.merge([self.constant_band]*3)

    def test_match_histogram(self):
        def luts_calculation(in_img, ref_img,
            in_mask=None, ref_mask=None):
            lut = numpy.ones(256, dtype=numpy.uint8)
            luts = [lut, 2*lut, 3*lut]
            return luts

        expected_img = cv2.merge(
            [self.constant_band,
             2*self.constant_band,
             3*self.constant_band])
        ret_img = hm.match_histogram(
            luts_calculation, self.sequence_img, self.constant_img)
        numpy.testing.assert_array_equal(ret_img, expected_img)

    def test_cdf_normalization_luts(self):
        sequence_to_compressed_luts = hm.cdf_normalization_luts(
            self.sequence_img,
            self.compressed_img)
        sequence_to_spread_lut = numpy.array(
            sorted(range(0, 256, 32) * 32),
            dtype=numpy.uint8)
        sequence_to_first_half_lut = numpy.array(
            sorted(range(0, 128) * 2),
            dtype=numpy.uint8)
        sequence_to_second_half_lut = numpy.array(
            sorted(range(128, 256) * 2),
            dtype=numpy.uint8)
        numpy.testing.assert_array_equal(sequence_to_compressed_luts,
            [sequence_to_spread_lut,
             sequence_to_first_half_lut,
             sequence_to_second_half_lut])

        compressed_to_sequence_luts = hm.cdf_normalization_luts(
            self.compressed_img,
            self.sequence_img)
        spread_to_sequence_lut = numpy.array(
            sorted(range(31, 256, 32) * 32),
            dtype=numpy.uint8)
        first_half_to_sequence_lut = numpy.array(
            range(1, 256, 2) + [255] * 128,
            dtype=numpy.uint8)
        second_half_to_sequence_lut = numpy.array(
            [0] * 128 + range(1, 256, 2),
            dtype=numpy.uint8)
        numpy.testing.assert_array_equal(compressed_to_sequence_luts,
            [spread_to_sequence_lut,
             first_half_to_sequence_lut,
             second_half_to_sequence_lut])

    def test__check_cdf(self):
        not_mono = numpy.array((0, 1, 0, 1))
        self.assertRaises(
            hm.CDFException, hm._check_cdf, not_mono)

        max_too_high = numpy.array((0, 0.5, 2))
        self.assertRaises(
            hm.CDFException, hm._check_cdf, max_too_high)

        max_too_low = numpy.array((0, 0.5, 0.999999))
        self.assertRaises(
            hm.CDFException, hm._check_cdf, max_too_low)

        hm._check_cdf(numpy.array((0, 0.5, 1)))


    def test_cdf_match_lut(self):
        # Intensity values at 3,4,5,6
        test_cdf = numpy.zeros((8))
        test_cdf[3] = 0.25
        test_cdf[4] = .5
        test_cdf[5] = 0.75
        test_cdf[6:] = 1.0

        # Intensity values at 1,2,3,4 (test minus 2)
        match_cdf = numpy.zeros((8))
        match_cdf[1] = 0.25
        match_cdf[2] = .5
        match_cdf[3] = 0.75
        match_cdf[4:] = 1
        logging.debug("match cdf: {}".format(match_cdf))

        # Test all values are mapped down by 2
        expected_lut = numpy.array([1, 1, 1, 1, 2, 3, 4, 4])
        lut = hm.cdf_match_lut(test_cdf, match_cdf)
        numpy.testing.assert_array_equal(lut, expected_lut)

        # Intensity values all at 4
        match_cdf = numpy.zeros((8))
        match_cdf[4:] = 1

        # Test all values are mapped to 4
        expected_lut = numpy.array([4, 4, 4, 4, 4, 4, 4, 4])
        lut = hm.cdf_match_lut(test_cdf, match_cdf)
        numpy.testing.assert_array_equal(lut, expected_lut)

        test_cdf = numpy.array([0, 9.99999881e-01])
        match_cdf = numpy.array([0, 1])
        self.assertRaises(hm.CDFException, hm.cdf_match_lut, test_cdf, match_cdf)


    def test_mean_std_luts(self):
        sequence_to_compressed_luts = hm.mean_std_luts(
            self.sequence_img,
            self.compressed_img)
        sequence_to_spread_lut = numpy.array(
            [0]*15 + range(0, 49) + range(48, 176) + range(175, 239),
            dtype=numpy.uint8)
        sequence_to_first_half_lut = numpy.array(
            [0] + sorted(range(0, 127) * 2) + [127],
            dtype=numpy.uint8)
        sequence_to_second_half_lut = numpy.array(
            [127] + sorted(range(128, 255) * 2) + [255],
            dtype=numpy.uint8)
        numpy.testing.assert_array_equal(sequence_to_compressed_luts,
            [sequence_to_spread_lut,
             sequence_to_first_half_lut,
             sequence_to_second_half_lut])

        compressed_to_sequence_luts = hm.mean_std_luts(
            self.compressed_img,
            self.sequence_img)
        spread_to_sequence_lut = numpy.array(
            range(14, 63) + range(64, 191) + range(192, 255) + [255] * 17,
            dtype=numpy.uint8)
        first_half_to_sequence_lut = numpy.array(
            range(0, 256, 2) + [255] * 128,
            dtype=numpy.uint8)
        second_half_to_sequence_lut = numpy.array(
            [0] * 128 + range(0, 256, 2),
            dtype=numpy.uint8)
        numpy.testing.assert_array_equal(compressed_to_sequence_luts,
            [spread_to_sequence_lut,
             first_half_to_sequence_lut,
             second_half_to_sequence_lut])


if __name__ == '__main__':
    unittest.main()
