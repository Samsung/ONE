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

import os.path
import unittest

# Python doesn't have const var but handle these as const
# It's meaning that DO NOT MODIFY these vars
CACHED_TEST_MODEL_DIR = "../../tests/scripts/models/cache"
TEST_MODEL_PATH = os.path.join(CACHED_TEST_MODEL_DIR, "convolution_test.tflite")


def Exist_CACHED_TEST_MODEL_DIR(dir):
    return os.path.exists(dir) and os.path.isdir(dir)


def Exist_TEST_MODEL_FILE(file):
    return os.path.exists(file) and os.path.isfile(file)


class Setup(unittest.TestCase):
    def test_Exist_CACHED_TEST_MODEL_DIR(self):
        model_dir = CACHED_TEST_MODEL_DIR
        self.assertTrue(Exist_CACHED_TEST_MODEL_DIR(model_dir))

    def test_Exist_TEST_MODEL_FILE(self):
        model_file = TEST_MODEL_PATH
        self.assertTrue(Exist_TEST_MODEL_FILE(model_file))


if __name__ == '__main__':
    unittest.main()
