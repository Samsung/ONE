#!/usr/bin/env python

# Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

import unittest
from printer.string_builder import *


class StringBuilderTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_ConvertBytesToHuman(self):
        SYMBOLS = ['B', 'K', 'M', 'G', 'T']
        format_str = "%(val)3.1f%(symb)s"

        bytes = -1
        self.assertEqual(ConvertBytesToHuman(bytes), 0)

        bytes = 1
        self.assertEqual(ConvertBytesToHuman(bytes),
                         format_str % dict(symb=SYMBOLS[0], val=(bytes)))

        bytes = 1024
        self.assertEqual(ConvertBytesToHuman(bytes),
                         format_str % dict(symb=SYMBOLS[1], val=(bytes / 1024)))

        bytes = 1024**2
        self.assertEqual(ConvertBytesToHuman(bytes),
                         format_str % dict(symb=SYMBOLS[2], val=(bytes / (1024**2))))

        bytes = 1024**3
        self.assertEqual(ConvertBytesToHuman(bytes),
                         format_str % dict(symb=SYMBOLS[3], val=(bytes / (1024**3))))

        bytes = 1024**4
        self.assertEqual(ConvertBytesToHuman(bytes),
                         format_str % dict(symb=SYMBOLS[4], val=(bytes / (1024**4))))

    # TODO: More tests


if __name__ == '__main__':
    unittest.main()
