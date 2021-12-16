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
from ir.operator import Operator


# Test the only getter/setter
class OperatorTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_index(self):
        op = Operator()
        op.index = 1000
        self.assertEqual(op.index, 1000)

    def test_inputs(self):
        op = Operator()
        t0 = Tensor()
        t0.index = 0
        t1 = Tensor()
        t1.index = 1
        op.inputs = [t0, t1]
        self.assertEqual(len(op.inputs), 2)
        self.assertEqual(op.inputs[0], t0)
        self.assertEqual(op.inputs[1], t1)

    def test_outputs(self):
        op = Operator()
        t0 = Tensor()
        t0.index = 0
        t1 = Tensor()
        t1.index = 1
        op.outputs = [t0, t1]
        self.assertEqual(len(op.outputs), 2)
        self.assertEqual(op.outputs[0], t0)
        self.assertEqual(op.outputs[1], t1)

    def test_op_name(self):
        op = Operator()
        op.op_name = "ADD"
        self.assertEqual(op.op_name, "ADD")

    def test_activation(self):
        op = Operator()
        op.activation = "Tanh"
        self.assertEqual(op.activation, "Tanh")

    def test_options(self):
        op = Operator()
        op.options = "Options ..."
        self.assertEqual(op.options, "Options ...")


if __name__ == '__main__':
    unittest.main()
