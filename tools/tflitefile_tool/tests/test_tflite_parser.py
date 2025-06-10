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
import tflite.Model
from parser.tflite.tflite_parser import TFLiteParser, TFLiteSubgraphParser
from .test_setup import TEST_MODEL_PATH


class TFLiteSubgraphParserTestCase(unittest.TestCase):
    def setUp(self):
        self.model_file = open(TEST_MODEL_PATH, 'rb')

    def tearDown(self):
        self.model_file.close()

    def test_Parse(self):
        buf = bytearray(self.model_file.read())
        tf_model = tflite.Model.Model.GetRootAsModel(buf, 0)
        for subgraph_index in range(tf_model.SubgraphsLength()):
            tf_subgraph = tf_model.Subgraphs(subgraph_index)
            subg_parser = TFLiteSubgraphParser(tf_model, subgraph_index)
            subg = subg_parser.Parse()
            self.assertEqual(subg.index, subgraph_index)
            self.assertEqual(len(subg.inputs), tf_subgraph.InputsLength())
            self.assertEqual(len(subg.outputs), tf_subgraph.OutputsLength())
            # if there is optional tensors, this assert could be wrong
            self.assertEqual(len(subg.tensors_map.keys()), tf_subgraph.TensorsLength())
            self.assertEqual(len(subg.operators_map.keys()),
                             tf_subgraph.OperatorsLength())
            # because TEST_MODEL_PATH has an op(ADD)
            self.assertEqual(len(subg.optypes_map.keys()), tf_subgraph.OperatorsLength())


class TFLiteParserTestCase(unittest.TestCase):
    def setUp(self):
        self.model_file = open(TEST_MODEL_PATH, 'rb')
        self.parser = TFLiteParser(self.model_file)

    def tearDown(self):
        self.model_file.close()

    def test_Parse(self):
        subg_list = self.parser.Parse()
        self.assertIsNotNone(subg_list)
        self.assertEqual(len(subg_list), 1)


if __name__ == '__main__':
    unittest.main()
