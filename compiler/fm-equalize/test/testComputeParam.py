#!/usr/bin/env python
# Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''Test fmelib.ComputeParam module'''

import unittest

import numpy as np

from fmelib.ComputeParam import _channelwiseMinMax


class ComputeParamTest(unittest.TestCase):
    def test_channelwiseMinMax(self):
        tensors = []
        # Channel-wise distribution
        # First channel: [1, 4, 7] min: 1, max: 7
        # Second channel: [2, 5, 8] min: 2, max: 8
        # Third channel: [3, 6, 9] min: 3, max: 9
        tensors.append(np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]))
        channel_wise_min, channel_wise_max = _channelwiseMinMax(tensors, 3)
        self.assertEqual(channel_wise_min, [1, 2, 3])
        self.assertEqual(channel_wise_max, [7, 8, 9])


if __name__ == '__main__':
    unittest.main()
