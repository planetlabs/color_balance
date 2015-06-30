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

from color_balance import mask


class Tests(unittest.TestCase):


    def test_create_mask(self):

        test_band = np.array([
            [1, 2],
            [3, 4]
        ], dtype=np.uint8)

        # Mask should be the same shape as band with all locations unmasked
        test_mask = mask.create_mask(test_band)
        expected_mask = np.array([
            [255, 255],
            [255, 255]
        ], dtype=np.uint8)
    
        np.testing.assert_array_equal(test_mask, expected_mask)

        # Mask should be the same shape as band with location corresponding
        # to 3 masked
        test_mask = mask.create_mask(test_band, value=3)
        expected_mask = np.array([
            [255, 255],
            [0, 255]
        ], dtype=np.uint8)
    
        np.testing.assert_array_equal(test_mask, expected_mask)

        # Mask should be the same shape as band with all locations unmasked
        # because the band contains no pixels with intensity equal to 'value'
        test_mask = mask.create_mask(test_band, value=6)
        expected_mask = np.array([
            [255, 255],
            [255, 255]
        ], dtype=np.uint8)
    
        np.testing.assert_array_equal(test_mask, expected_mask)


    def test_combine_masks(self):

        mask1 = np.array([
            [0, 255],
            [255, 255]
        ], dtype=np.uint8)

        mask2 = np.array([
            [255, 0],
            [255, 255]
        ], dtype=np.uint8)

        test_mask = mask.combine_masks([mask1, mask2])

        expected_mask = np.array([
            [0, 0],
            [255, 255]
        ], dtype=np.uint8)

        np.testing.assert_array_equal(test_mask, expected_mask)


    def test_map_masked(self):
    
        test_band = np.array([
            [1, 1],
            [1, 1]
        ], dtype=np.uint8)

        test_mask = np.array([
            [255, 0],
            [255, 0]
        ], dtype=np.uint8)

        # Test appropriate entries were mapped
        expected_band = np.array([
            [1, 7],
            [1, 7]
        ], dtype=np.uint8)
        masked_band = mask.map_masked(test_band, test_mask, value=7)
    
        np.testing.assert_array_equal(masked_band, expected_band)

        # Test default masked value
        expected_band = np.array([
            [1, 0],
            [1, 0]
        ], dtype=np.uint8)
        masked_band = mask.map_masked(test_band, test_mask)

        np.testing.assert_array_equal(masked_band, expected_band)


    def test_get_mask_percent(self):
    
        # 4 of 4 values are masked
        test_mask = np.array([
            [0, 0],
            [0, 0]
        ], dtype=np.uint8)
        expected = 100
        test_percent = mask.get_mask_percent(test_mask)
        assert expected == test_percent
    
        # 3 of 4 values are masked
        test_mask = np.array([
            [0, 0],
            [255, 0]
        ], dtype=np.uint8)
        expected = 75
        test_percent = mask.get_mask_percent(test_mask)
        assert expected == test_percent

        # 0 of 4 values are masked
        test_mask = np.array([
            [255, 255],
            [255, 255]
        ], dtype=np.uint8)
        expected = 0
        test_percent = mask.get_mask_percent(test_mask)
        assert expected == test_percent
