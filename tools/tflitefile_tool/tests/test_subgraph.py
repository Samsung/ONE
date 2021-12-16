#!/usr/bin/env python

# Csubgyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a csubgy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from ir.subgraph import Subgraph
from ir.operator import Operator
from ir.tensor import Tensor


# Test the only getter/setter
class SubgraphTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_index(self):
        subg = Subgraph()
        subg.index = 1000
        self.assertEqual(subg.index, 1000)

    def test_inputs(self):
        subg = Subgraph()
        t0 = Tensor()
        t0.index = 0
        t1 = Tensor()
        t1.index = 1
        subg.inputs = [t0, t1]
        self.assertEqual(len(subg.inputs), 2)
        self.assertEqual(subg.inputs[0], t0)
        self.assertEqual(subg.inputs[0].index, 0)
        self.assertEqual(subg.inputs[1], t1)
        self.assertEqual(subg.inputs[1].index, 1)

    def test_outputs(self):
        subg = Subgraph()
        t0 = Tensor()
        t0.index = 0
        t1 = Tensor()
        t1.index = 1
        subg.outputs = [t0, t1]
        self.assertEqual(len(subg.outputs), 2)
        self.assertEqual(subg.outputs[0], t0)
        self.assertEqual(subg.outputs[0].index, 0)
        self.assertEqual(subg.outputs[1], t1)
        self.assertEqual(subg.outputs[1].index, 1)

    def test_subg_name(self):
        subg = Subgraph()
        subg.subg_name = "SUBGRAPH_0"
        self.assertEqual(subg.subg_name, "SUBGRAPH_0")

    def test_model_name(self):
        subg = Subgraph()
        subg.model_name = "SUBGRAPH_0"
        self.assertEqual(subg.model_name, "SUBGRAPH_0")

    def test_tensors_map(self):
        subg = Subgraph()
        t0 = Tensor()
        t0.index = 0
        t1 = Tensor()
        t1.index = 1
        subg.tensors_map[t0.index] = t0
        subg.tensors_map[t1.index] = t1
        self.assertEqual(len(subg.tensors_map.keys()), 2)
        self.assertEqual(subg.tensors_map[t0.index], t0)
        self.assertEqual(subg.tensors_map[t1.index], t1)

    def test_operators_map(self):
        subg = Subgraph()
        op0 = Operator()
        op0.index = 0
        op0.op_name = "ADD"
        op1 = Operator()
        op1.index = 1
        op1.op_name = "SUB"
        subg.operators_map[op0.index] = op0
        subg.operators_map[op1.index] = op1
        self.assertEqual(len(subg.operators_map.keys()), 2)
        self.assertEqual(subg.operators_map[op0.index], op0)
        self.assertEqual(subg.operators_map[op1.index], op1)

    def test_optypes_map(self):
        subg = Subgraph()
        op0 = Operator()
        op0.index = 0
        op0.op_name = "ADD"
        op1 = Operator()
        op1.index = 1
        op1.op_name = "SUB"
        op2 = Operator()
        op2.index = 2
        op2.op_name = "SUB"

        subg.optypes_map[op0.op_name] = op0
        subg.optypes_map[op1.op_name] = op1
        subg.optypes_map[op2.op_name] = op2

        self.assertEqual(len(subg.optypes_map.keys()), 2)
        self.assertEqual(len(subg.optypes_map[op0.op_name]), 1)
        self.assertEqual(len(subg.optypes_map[op2.op_name]), 2)


if __name__ == '__main__':
    unittest.main()
