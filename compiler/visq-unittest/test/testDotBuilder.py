# Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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
"""Test visqlib.DotBuilder module"""

import unittest
import pydot

from visqlib.DotBuilder import DotBuilder
from test.Resources import fp32_model_dir


class VisqDotBuilderTest(unittest.TestCase):
    def test_dot_builder_wrong_input_file(self):
        self.assertRaises(FileNotFoundError, DotBuilder, "wrong", "wrong", "wrong",
                          "wrong")

    def test_dot_builder(self):
        test_colors = [{"b": 0, "e": 0.5, "c": "green"}, {"b": 0.5, "e": 1, "c": "red"}]
        test_qerror_map = dict()
        test_qerror_map["ofm"] = 0.1
        builder = DotBuilder(fp32_model_dir + "/Add_000.circle", "Add_000.dot", "MPEIR",
                             test_colors)
        builder.save(test_qerror_map)

        graph = pydot.graph_from_dot_file("Add_000.dot")[0]
        # Why 1? 0 is output
        ofm_node = graph.get_node("\"ofm\"")[1]
        self.assertEqual("green", ofm_node.get_fillcolor())


if __name__ == "__main__":
    unittest.main()
