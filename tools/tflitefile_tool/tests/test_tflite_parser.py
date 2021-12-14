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
from parser.tflite_parser import TFLiteParser
from .test_setup import TEST_MODEL_PATH


class TFLiteParserTestCase(unittest.TestCase):
    def setUp(self):
        self.model_file = open(TEST_MODEL_PATH, 'rb')
        self.parser = TFLiteParser(self.model_file)

    def tearDown(self):
        self.model_file.close()

    def test_Parse(self):
        (subg_list, stats) = self.parser.Parse()
        self.assertIsNotNone(subg_list)
        self.assertEqual(len(subg_list), 1)
        self.assertIsNotNone(stats)


if __name__ == '__main__':
    unittest.main()
