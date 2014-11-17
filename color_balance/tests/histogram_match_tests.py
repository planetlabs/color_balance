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

import numpy
import cv2

from color_balance import histogram_match as hm


class HistogramMatchTests(unittest.TestCase):
    def setUp(self):
        band = numpy.array([0, 1, 2, 3], dtype=numpy.uint8)
        self.min_sequence_img = cv2.merge([band, band, band])

        band = numpy.array([252, 253, 254, 255], dtype=numpy.uint8)
        self.max_sequence_img = cv2.merge([band, band, band])

        band = numpy.array([0, 50, 200, 255], dtype=numpy.uint8)
        self.min_max_spread_img = cv2.merge([band, band, band])

        band = numpy.array([0, 0, 0, 0], dtype=numpy.uint8)
        self.min_constant_img = cv2.merge([band, band, band])

        band = numpy.array([255, 255, 255, 255], dtype=numpy.uint8)
        self.max_constant_img = cv2.merge([band, band, band])

        self.unique_input_img = cv2.merge((
            numpy.array([0, 1, 2, 3], dtype=numpy.uint8),
            numpy.array([50, 100, 150, 200], dtype=numpy.uint8),
            numpy.array([252, 253, 254, 255], dtype=numpy.uint8),
            ))

        self.repeating_input_img = cv2.merge((
            numpy.array([0, 1, 1, 0], dtype=numpy.uint8),
            numpy.array([100, 100, 100, 100], dtype=numpy.uint8),
            numpy.array([200, 200, 225, 255], dtype=numpy.uint8),
            ))

    def test_match_histogram_cdf_norm_unique_input_values(self):
        test_img = hm.match_histogram(
            self.unique_input_img,
            self.min_sequence_img,
            hist_match_method=hm.HIST_MATCH_METHOD.CDF_NORM)
        numpy.testing.assert_array_equal(test_img, self.min_sequence_img)

        test_img = hm.match_histogram(
            self.unique_input_img,
            self.max_sequence_img,
            hist_match_method=hm.HIST_MATCH_METHOD.CDF_NORM)
        numpy.testing.assert_array_equal(test_img, self.max_sequence_img)

        test_img = hm.match_histogram(
            self.unique_input_img,
            self.min_constant_img,
            hist_match_method=hm.HIST_MATCH_METHOD.CDF_NORM)
        numpy.testing.assert_array_equal(test_img, self.min_constant_img)

        test_img = hm.match_histogram(
            self.unique_input_img,
            self.max_constant_img,
            hist_match_method=hm.HIST_MATCH_METHOD.CDF_NORM)
        numpy.testing.assert_array_equal(test_img, self.max_constant_img)

    def test_match_histogram_cdf_norm_repeating_input_values(self):
        # Can't map repeated values to all unique values so result
        # will deviate from reference image somewhat
        test_img = hm.match_histogram(
            self.repeating_input_img,
            self.min_sequence_img,
            hist_match_method=hm.HIST_MATCH_METHOD.CDF_NORM)
        expected_img = cv2.merge((
            numpy.array([1, 3, 3, 1], dtype=numpy.uint8),
            numpy.array([3, 3, 3, 3], dtype=numpy.uint8),
            numpy.array([1, 1, 2, 3], dtype=numpy.uint8),
            ))
        numpy.testing.assert_array_equal(test_img, expected_img)

        test_img = hm.match_histogram(
            self.repeating_input_img,
            self.max_sequence_img,
            hist_match_method=hm.HIST_MATCH_METHOD.CDF_NORM)       
        expected_img = cv2.merge((
            numpy.array([253, 255, 255, 253], dtype=numpy.uint8),
            numpy.array([255, 255, 255, 255], dtype=numpy.uint8),
            numpy.array([253, 253, 254, 255], dtype=numpy.uint8),
            ))
        numpy.testing.assert_array_equal(test_img, expected_img)

        # Can map repeated values to constant values
        test_img = hm.match_histogram(
            self.repeating_input_img,
            self.min_constant_img,
            hist_match_method=hm.HIST_MATCH_METHOD.CDF_NORM)
        numpy.testing.assert_array_equal(test_img, self.min_constant_img)

        test_img = hm.match_histogram(
            self.repeating_input_img,
            self.max_constant_img,
            hist_match_method=hm.HIST_MATCH_METHOD.CDF_NORM)
        numpy.testing.assert_array_equal(test_img, self.max_constant_img)

    def test_match_histogram_mean_std_unique_input_values(self):
        test_img = hm.match_histogram(
            self.unique_input_img,
            self.min_sequence_img,
            hist_match_method=hm.HIST_MATCH_METHOD.MEAN_STD)
        numpy.testing.assert_array_equal(test_img, self.min_sequence_img)

        test_img = hm.match_histogram(
            self.unique_input_img,
            self.max_sequence_img,
            hist_match_method=hm.HIST_MATCH_METHOD.MEAN_STD)
        numpy.testing.assert_array_equal(test_img, self.max_sequence_img)

        test_img = hm.match_histogram(
            self.unique_input_img,
            self.min_constant_img,
            hist_match_method=hm.HIST_MATCH_METHOD.MEAN_STD)
        numpy.testing.assert_array_equal(test_img, self.min_constant_img)

        test_img = hm.match_histogram(
            self.unique_input_img,
            self.max_constant_img,
            hist_match_method=hm.HIST_MATCH_METHOD.MEAN_STD)
        numpy.testing.assert_array_equal(test_img, self.max_constant_img)

    def test_match_histogram_mean_std_repeating_input_values(self):
        # Can't map repeated values to all unique values so result
        # will deviate from reference image somewhat
        test_img = hm.match_histogram(
            self.repeating_input_img,
            self.min_sequence_img,
            hist_match_method=hm.HIST_MATCH_METHOD.MEAN_STD)
        expected_img = cv2.merge((
            numpy.array([0, 2, 2, 0], dtype=numpy.uint8),
            numpy.array([1, 1, 1, 1], dtype=numpy.uint8),
            numpy.array([0, 0, 1, 3], dtype=numpy.uint8),
            ))
        numpy.testing.assert_array_equal(
            test_img.ravel(), expected_img.ravel())

        test_img = hm.match_histogram(
            self.repeating_input_img,
            self.max_sequence_img,
            hist_match_method=hm.HIST_MATCH_METHOD.MEAN_STD)       
        expected_img = cv2.merge((
            numpy.array([252, 254, 254, 252], dtype=numpy.uint8),
            numpy.array([253, 253, 253, 253], dtype=numpy.uint8),
            numpy.array([252, 252, 253, 255], dtype=numpy.uint8),
            ))
        numpy.testing.assert_array_equal(
            test_img.ravel(), expected_img.ravel())

        # Can map repeated values to constant values
        test_img = hm.match_histogram(
            self.repeating_input_img,
            self.min_constant_img,
            hist_match_method=hm.HIST_MATCH_METHOD.MEAN_STD)
        numpy.testing.assert_array_equal(test_img, self.min_constant_img)

        test_img = hm.match_histogram(
            self.repeating_input_img,
            self.max_constant_img,
            hist_match_method=hm.HIST_MATCH_METHOD.MEAN_STD)
        numpy.testing.assert_array_equal(test_img, self.max_constant_img)    

class HistogramDistanceTests(unittest.TestCase):
    def setUp(self):
        self.uniform_histogram = numpy.array(
            [1, 1, 1, 1, 1, 1], dtype=numpy.uint8)
        self.uniform_first_half_histogram = numpy.array(
            [2, 2, 2, 0, 0, 0], dtype=numpy.uint8)
        self.uniform_second_half_histogram = numpy.array(
            [0, 0, 0, 2, 2, 2], dtype=numpy.uint8)
        self.uniform_middle_histogram = numpy.array(
            [0, 2, 2, 2, 0, 0], dtype=numpy.uint8)
        self.downsampled_uniform_histogram = numpy.array(
            [2, 0, 2, 0, 2, 0], dtype=numpy.uint8)

    def run_histogram_distance_calculations(self, distance):
        distances = {}
        distances['same_histogram_distance'] = hm.histogram_distance(
            self.uniform_histogram,
            self.uniform_histogram, 
            distance=distance)

        distances['downsampled_histogram_distance'] = hm.histogram_distance(
            self.uniform_histogram,
            self.downsampled_uniform_histogram, 
            distance=distance)        

        distances['compressed_histogram_distance'] = hm.histogram_distance(
            self.uniform_histogram,
            self.uniform_first_half_histogram, 
            distance=distance)

        distances['translated_histogram_distance'] = hm.histogram_distance(
            self.uniform_first_half_histogram,
            self.uniform_middle_histogram, 
            distance=distance)

        distances['nonoverlapping_histogram_distance'] = hm.histogram_distance(
            self.uniform_first_half_histogram,
            self.uniform_second_half_histogram, 
            distance=distance)

        return distances

    def test_histogram_distance(self):
        chi_squared_distances = self.run_histogram_distance_calculations(
            hm.HIST_DISTANCE.CHI_SQUARED)

        self.assertAlmostEquals(
            chi_squared_distances['same_histogram_distance'], 0.0, 1)
        self.assertAlmostEquals(
            chi_squared_distances['downsampled_histogram_distance'], 6.0, 1)
        self.assertAlmostEquals(
            chi_squared_distances['compressed_histogram_distance'], 6.0, 1)
        self.assertAlmostEquals(
            chi_squared_distances['translated_histogram_distance'], 2.0, 1)
        self.assertAlmostEquals(
            chi_squared_distances['nonoverlapping_histogram_distance'], 6.0, 1)

        correlation_distances = self.run_histogram_distance_calculations(
            hm.HIST_DISTANCE.CORRELATION)

        self.assertAlmostEquals(
            correlation_distances['same_histogram_distance'], 1.0, 0)
        self.assertAlmostEquals(
            correlation_distances['downsampled_histogram_distance'], 1.0, 1)
        self.assertAlmostEquals(
            correlation_distances['compressed_histogram_distance'], 1.0, 1)
        self.assertAlmostEquals(
            correlation_distances['translated_histogram_distance'], 0.3, 1)
        self.assertAlmostEquals(
            correlation_distances['nonoverlapping_histogram_distance'], -1.0, 1)

        jensen_shannon_distances = self.run_histogram_distance_calculations(
            hm.HIST_DISTANCE.JENSEN_SHANNON)

        self.assertAlmostEquals(
            jensen_shannon_distances['same_histogram_distance'], -12.0, 1)
        self.assertAlmostEquals(
            jensen_shannon_distances['downsampled_histogram_distance'], -8.26, 1)
        self.assertAlmostEquals(
            jensen_shannon_distances['compressed_histogram_distance'], -8.26, 1)
        self.assertAlmostEquals(
            jensen_shannon_distances['translated_histogram_distance'], -8.0, 1)
        self.assertAlmostEquals(
            jensen_shannon_distances['nonoverlapping_histogram_distance'], 0.0, 1)

if __name__ == '__main__':
    unittest.main()
