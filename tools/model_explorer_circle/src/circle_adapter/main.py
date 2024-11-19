#!/usr/bin/env python3

# Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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
"""Model Explorer adapter for Circle models."""

from typing import Dict
from model_explorer import Adapter, AdapterMetadata, ModelExplorerGraphs, graph_builder
from circle_adapter import circle_schema_generated as circle_schema


class CircleAdapter(Adapter):
    """Adapter class for Circle models."""
    metadata = AdapterMetadata(id='circle-adapter',
                               name='Circle adapter',
                               description='Circle adapter!',
                               fileExts=['circle'])

    def __init__(self):
        super().__init__()
        self.model = None
        self.dict_opcode_to_name = {
            v: k
            for k, v in circle_schema.BuiltinOperator.__dict__.items()
        }

    def load_model(self, model_path: str) -> None:
        """Load the model from the given path."""
        with open(model_path, 'rb') as fp:
            model_ = circle_schema.Model.GetRootAsModel(fp.read(), 0)

        self.model = circle_schema.ModelT.InitFromObj(model_)

    def opcode_to_name(self, opcode: int) -> str:
        """Convert the opcode to its name."""
        return self.dict_opcode_to_name[opcode]

    def build_graph(self, me_graph: graph_builder.Graph) -> None:
        """Build the graph using the model."""

        sub_graph = self.model.subgraphs[0]

        # Create Input nodes
        input_id = len(sub_graph.operators)
        me_node = graph_builder.GraphNode(id=f'{input_id}',
                                          label="GraphInputs",
                                          namespace="GraphInputs")
        me_graph.nodes.append(me_node)

        # Create operator nodes
        for idx, op in enumerate(sub_graph.operators):
            name = self.opcode_to_name(
                self.model.operatorCodes[op.opcodeIndex].builtinCode)
            me_node = graph_builder.GraphNode(id=f'{idx}', label=name)
            me_graph.nodes.append(me_node)

        # Create Output nodes
        me_node = graph_builder.GraphNode(id=f'{len(me_graph.nodes)}',
                                          label="GraphOutputs",
                                          namespace="GraphOutputs")
        me_graph.nodes.append(me_node)

    def convert(self, model_path: str, settings: Dict) -> ModelExplorerGraphs:
        """Convert the model to a set of graphs."""
        graph = graph_builder.Graph(id='main')

        self.load_model(model_path)
        self.build_graph(graph)

        return {'graphs': [graph]}
