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
# TODO: Do not use this module
import tflite.Model
import tflite.SubGraph
from ir import graph_stats
from .subgraph_parser import SubgraphParser


class TFLiteParser(object):
    def __init__(self, model_file):
        self.model_file = model_file

    def Parse(self):
        # Generate Model: top structure of tflite model file
        buf = self.model_file.read()
        buf = bytearray(buf)
        tf_model = tflite.Model.Model.GetRootAsModel(buf, 0)

        stats = graph_stats.GraphStats()
        # Model file can have many models
        subg_list = list()
        for subgraph_index in range(tf_model.SubgraphsLength()):
            tf_subgraph = tf_model.Subgraphs(subgraph_index)
            model_name = "#{0} {1}".format(subgraph_index, tf_subgraph.Name())
            # 0th subgraph is main subgraph
            if (subgraph_index == 0):
                model_name += " (MAIN)"

            # Parse Subgraphs
            subg_parser = SubgraphParser(tf_model, tf_subgraph)
            subg_parser.Parse()

            stats += graph_stats.CalcGraphStats(subg_parser)

            subg = (model_name, subg_parser)
            subg_list.append(subg)

        # Validate
        assert subg_list is not None
        assert len(subg_list) > 0
        assert stats is not None

        return (subg_list, stats)
