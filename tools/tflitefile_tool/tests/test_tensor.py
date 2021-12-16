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
from ir.tensor import Tensor


# Test the only getter/setter
class TensorTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_index(self):
        t = Tensor()
        t.index = 1000
        self.assertEqual(t.index, 1000)

    def test_tensor_name(self):
        t = Tensor()
        t.tensor_name = "input"
        self.assertEqual(t.tensor_name, "input")

    def test_buffer(self):
        t = Tensor()
        o = object()
        t.buffer = o
        self.assertEqual(t.buffer, o)

    def test_type_name(self):
        t = Tensor()
        t.type_name = "FLOAT32"
        self.assertEqual(t.type_name, "FLOAT32")

    def test_shape(self):
        t = Tensor()
        t.shape = [1, 2, 3, 4]
        self.assertEqual(t.shape, [1, 2, 3, 4])

    def test_memory_size(self):
        t = Tensor()
        t.memory_size = 1000
        self.assertEqual(t.memory_size, 1000)


if __name__ == '__main__':
    unittest.main()
