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
'''Test visqlib.Util module'''

import unittest

from visqlib.Util import to_filename
from visqlib.Util import valid_attr
from visqlib.Util import pretty_float


class VisqUtilTest(unittest.TestCase):
    def test_to_filename(self):
        data = 'abc/d/e'
        self.assertEqual('abc_d_e', to_filename(data))

        long_data = 'x' * 300
        self.assertEqual('x' * 255, to_filename(long_data))

    def test_valid_attr(self):
        class Test:
            def __init__(self):
                self.a = 'a'

        test = Test()
        self.assertTrue(valid_attr(test, 'a'))
        self.assertFalse(valid_attr(test, 'b'))

    def test_pretty_float(self):
        test_configs = [0.123456, 12.3456, [0.123456], {'test': [0.123456]}]
        three_digits_ans = [0.123, 12.346, [0.123], {'test': [0.123]}]
        for test_data, ans in zip(test_configs, three_digits_ans):
            res = pretty_float(test_data, ndigits=3)
            self.assertEqual(res, ans)

        test_configs = [0.123456, 12.3456, [0.123456], {'test': [0.123456]}]
        four_digits_ans = [0.1235, 12.3456, [0.1235], {'test': [0.1235]}]
        for test_data, ans in zip(test_configs, four_digits_ans):
            res = pretty_float(test_data, ndigits=4)
            self.assertEqual(res, ans)


if __name__ == '__main__':
    unittest.main()
