import unittest

import numpy
import cv2

from color_balance import histogram_distance as hd


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

    def test_image_distances(self):
        def distance_function(hist1, hist2):
            return 5

        band = numpy.ones((2,2), dtype=numpy.uint8)
        im1 = cv2.merge([band]*3)
        hd.image_distances(distance_function, im1, im1)        

    def test_chi_squared(self):
        same_histogram_distance = hd.chi_squared(
            self.uniform_histogram,
            self.uniform_histogram)
        self.assertAlmostEquals(same_histogram_distance, 0.0, 1)

        downsampled_histogram_distance = hd.chi_squared(
            self.uniform_histogram,
            self.downsampled_uniform_histogram)
        self.assertAlmostEquals(downsampled_histogram_distance, 6.0, 1)

        compressed_histogram_distance = hd.chi_squared(
            self.uniform_histogram,
            self.uniform_first_half_histogram)
        self.assertAlmostEquals(compressed_histogram_distance, 6.0, 1)

        translated_histogram_distance = hd.chi_squared(
            self.uniform_first_half_histogram,
            self.uniform_middle_histogram)
        self.assertAlmostEquals(translated_histogram_distance, 2.0, 1)

        nonoverlapping_histogram_distance = hd.chi_squared(
            self.uniform_first_half_histogram,
            self.uniform_second_half_histogram)
        self.assertAlmostEquals(nonoverlapping_histogram_distance, 6.0, 1)

    def test_correlation(self):
        same_histogram_distance = hd.correlation(
            self.uniform_histogram,
            self.uniform_histogram)
        self.assertAlmostEquals(same_histogram_distance, 1.0, 0)

        downsampled_histogram_distance = hd.correlation(
            self.uniform_histogram,
            self.downsampled_uniform_histogram)
        self.assertAlmostEquals(downsampled_histogram_distance, 1.0, 1)

        compressed_histogram_distance = hd.correlation(
            self.uniform_histogram,
            self.uniform_first_half_histogram)
        self.assertAlmostEquals(compressed_histogram_distance, 1.0, 1)

        translated_histogram_distance = hd.correlation(
            self.uniform_first_half_histogram,
            self.uniform_middle_histogram)
        self.assertAlmostEquals(translated_histogram_distance, 0.3, 1)

        nonoverlapping_histogram_distance = hd.correlation(
            self.uniform_first_half_histogram,
            self.uniform_second_half_histogram)
        self.assertAlmostEquals(nonoverlapping_histogram_distance, -1.0, 1)

    def test_jensen_shannon(self):
        same_histogram_distance = hd.jensen_shannon(
            self.uniform_histogram,
            self.uniform_histogram)
        self.assertAlmostEquals(same_histogram_distance, -12.0, 0)

        downsampled_histogram_distance = hd.jensen_shannon(
            self.uniform_histogram,
            self.downsampled_uniform_histogram)
        self.assertAlmostEquals(downsampled_histogram_distance, -8.26, 1)

        compressed_histogram_distance = hd.jensen_shannon(
            self.uniform_histogram,
            self.uniform_first_half_histogram)
        self.assertAlmostEquals(compressed_histogram_distance, -8.26, 1)

        translated_histogram_distance = hd.jensen_shannon(
            self.uniform_first_half_histogram,
            self.uniform_middle_histogram)
        self.assertAlmostEquals(translated_histogram_distance, -8.0, 1)

        nonoverlapping_histogram_distance = hd.jensen_shannon(
            self.uniform_first_half_histogram,
            self.uniform_second_half_histogram)
        self.assertAlmostEquals(nonoverlapping_histogram_distance, 0.0, 1)


if __name__ == '__main__':
    unittest.main()
