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

from typing import Dict, Optional
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
        # tensor_id -> node_id/output_id
        self.map_tensor_to_source = {}

    def load_model(self, model_path: str) -> None:
        """Load the model from the given path."""
        with open(model_path, 'rb') as fp:
            model_ = circle_schema.Model.GetRootAsModel(fp.read(), 0)

        self.model = circle_schema.ModelT.InitFromObj(model_)

    def opcode_to_name(self, opcode: int) -> str:
        """Convert the opcode to its name."""
        return self.dict_opcode_to_name[opcode]

    def set_source_of(self, tensor_id: int, source_id: int, output_id: int) -> None:
        """Set the source of the tensor."""
        self.map_tensor_to_source[tensor_id] = f'{source_id}/{output_id}'

    def get_source_of(self, tensor_id: int) -> Optional[str]:
        """Get the source of the tensor."""
        return self.map_tensor_to_source.get(tensor_id)

    def add_incoming_edge(self, me_node: graph_builder.GraphNode, tensor_id: int,
                          input_id: int) -> None:
        """Add incoming edge to the given node."""
        source = self.get_source_of(tensor_id)
        if source is not None:
            sid, soid = source.split('/')
            me_node.incomingEdges.append(
                graph_builder.IncomingEdge(sourceNodeId=sid,
                                           sourceNodeOutputId=soid,
                                           targetNodeInputId=f'{input_id}'))

    def build_graph(self, me_graph: graph_builder.Graph) -> None:
        """Build the graph using the model."""

        sub_graph = self.model.subgraphs[0]

        # Create Input nodes
        input_id = len(sub_graph.operators)
        me_node = graph_builder.GraphNode(id=f'{input_id}',
                                          label="GraphInputs",
                                          namespace="GraphInputs")
        me_graph.nodes.append(me_node)

        # Map source and output tensors of GraphInputs
        for i, tensor_id in enumerate(sub_graph.inputs):
            self.set_source_of(tensor_id=tensor_id, source_id=input_id, output_id=i)

        # Map source and output tensors of operators
        for op_id, op in enumerate(sub_graph.operators):
            for i, tensor_id in enumerate(op.outputs):
                self.set_source_of(tensor_id=tensor_id, source_id=op_id, output_id=i)

        # Create pseudo const node for orphan tensors (= const tensors)
        for tensor_id, tensor in enumerate(sub_graph.tensors):
            if (self.get_source_of(tensor_id)) is None:
                me_node = graph_builder.GraphNode(id=f'{input_id + tensor_id}',
                                                  label='pseudo_const',
                                                  namespace=tensor.name.decode('utf-8'))
                me_graph.nodes.append(me_node)
                # Map source and output tensors of const tensor
                self.set_source_of(tensor_id=tensor_id,
                                   source_id=input_id + tensor_id,
                                   output_id=0)

        # Create operator nodes
        for idx, op in enumerate(sub_graph.operators):
            name = self.opcode_to_name(
                self.model.operatorCodes[op.opcodeIndex].builtinCode)
            me_node = graph_builder.GraphNode(id=f'{idx}', label=name)
            me_graph.nodes.append(me_node)
            # Connect edges from inputs to this operator node
            for i, tensor_id in enumerate(op.inputs):
                if tensor_id < 0:
                    continue
                self.add_incoming_edge(me_node=me_node, tensor_id=tensor_id, input_id=i)

        # Create Output nodes
        me_node = graph_builder.GraphNode(id=f'{len(me_graph.nodes)}',
                                          label="GraphOutputs",
                                          namespace="GraphOutputs")
        me_graph.nodes.append(me_node)
        # Connect edges from inputs to output node
        for i in sub_graph.outputs:
            self.add_incoming_edge(me_node=me_node, tensor_id=i, input_id=i)

    def convert(self, model_path: str, settings: Dict) -> ModelExplorerGraphs:
        """Convert the model to a set of graphs."""
        graph = graph_builder.Graph(id='main')

        self.load_model(model_path)
        self.build_graph(graph)

        return {'graphs': [graph]}
