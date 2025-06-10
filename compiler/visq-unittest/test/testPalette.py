# Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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
'''Test visqlib.Palette module'''

import unittest

from visqlib.Palette import YLORRD9Palette


class VisqPaletteTest(unittest.TestCase):
    def test_ylorrd9(self):
        min_test = [0.0, 0, -100, -100]
        max_test = [1.0, 500, 100, -10]

        for min_val, max_val in zip(min_test, max_test):
            palette = YLORRD9Palette(qerror_min=min_val, qerror_max=max_val)
            cs = palette.colorscheme()
            self.assertEqual(9, len(cs))

    def test_ylorrd9_wrong_minmax(self):
        min_test = [0.0, 10]
        max_test = [0.0, 0]

        for min_val, max_val in zip(min_test, max_test):
            # min must be less than max
            self.assertRaises(RuntimeError,
                              YLORRD9Palette,
                              qerror_min=min_val,
                              qerror_max=max_val)


if __name__ == '__main__':
    unittest.main()
