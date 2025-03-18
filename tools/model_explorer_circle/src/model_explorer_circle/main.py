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
from model_explorer_circle import circle_schema_generated as circle_schema
from model_explorer_circle import circle_metadata


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
        self.dict_tensor_type_to_string = {
            v: k
            for k, v in circle_schema.TensorType.__dict__.items()
        }
        # Number of elements to show in tensor values (default: 16)
        self.const_element_count_limit = 16
        self.graph = None
        self.input_args = circle_metadata.input_args_dict

    def load_model(self, model_path: str) -> None:
        """Load the model from the given path."""
        with open(model_path, 'rb') as fp:
            model_ = circle_schema.Model.GetRootAsModel(fp.read(), 0)

        self.model = circle_schema.ModelT.InitFromObj(model_)

    def get_opcode_name(self, opcode_index: int) -> str:
        """Convert the opcode to its name."""
        opcode = self.model.operatorCodes[opcode_index].deprecatedBuiltinCode
        if opcode == circle_schema.BuiltinOperator.CUSTOM:
            return self.model.operatorCodes[opcode_index].customCode.decode('utf-8')
        if opcode == circle_schema.BuiltinOperator.PLACEHOLDER_FOR_GREATER_OP_CODES:
            opcode = self.model.operatorCodes[opcode_index].builtinCode
            assert (opcode >= 127)
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

        metadata = graph_builder.MetadataItem(id=f'{input_id}')
        # Add input metadata of the node if it exists
        if self.input_args.get(me_node.label) is not None:
            arity = len(self.input_args[me_node.label])
            if input_id < arity:
                arg_name = self.input_args[me_node.label][input_id]
            else:
                # For variadic inputs, append index to the last argument name
                arg_name = self.input_args[me_node.label][arity - 1]
                # Update 1st argument name of variadic inputs when 2nd argument name is added
                if input_id == arity:
                    me_node.inputsMetadata[-1].attrs[0].value = arg_name + '[0]'
                arg_name += f'[{input_id - arity + 1}]'
            metadata.attrs.append(
                graph_builder.KeyValue(key='__tensor_tag', value=arg_name))

        # Quantization parameter (if exists)
        tensor = self.model.subgraphs[0].tensors[tensor_id]
        if tensor.quantization and tensor.quantization.scale is not None:
            quantparam = '['
            for i, scale in enumerate(tensor.quantization.scale):
                if i != 0:
                    quantparam += ', '
                # Show the most significant 6 digits of the scale
                scale = format(scale, '.6g')
                zp = tensor.quantization.zeroPoint[i]
                # If the type is larger than INT8, exponential notation will be used
                quantparam += f'{scale} * (q - {zp})'
            quantparam += ']'
            metadata.attrs.append(
                graph_builder.KeyValue(key='quantization', value=quantparam))

        if len(metadata.attrs) > 0:
            me_node.inputsMetadata.append(metadata)

    def add_tensor_value_attribute(self, me_node: graph_builder.GraphNode,
                                   tensor_id: int) -> None:
        """Prints a tensor with the specified number of elements"""
        tensor = self.model.subgraphs[0].tensors[tensor_id]
        buffer = self.model.buffers[tensor.buffer].data
        if buffer is None:
            return
        dtype = self.dict_tensor_type_to_string[tensor.type].lower()
        # Convert buffer into numpy array with the correct datatype
        if dtype in ['int4', 'uint4']:
            tensor_data = buffer.view(dtype[:-1] + '8')
            n_elements = buffer.size * 2
        else:
            tensor_data = buffer.view(dtype)
            n_elements = len(tensor_data)
        shape = list(tensor.shape)
        rank = len(shape) if shape else 1

        # Used to represent the number of elements in each dimension
        counter = [0] * rank
        # How many elements are we going to print?
        n_elements = min(self.const_element_count_limit, n_elements)
        # How many brackets are currently opened?
        open_brackets = 0
        # Ouput string
        tensor_data_str = ''

        # Form a string representation of the tensor data
        for i in range(n_elements):
            if i != 0:
                tensor_data_str += ', '
            tensor_data_str += '[' * (rank - open_brackets)
            open_brackets = rank
            if dtype in ['int4', 'uint4']:
                val_4b = (tensor_data[i // 2] >> 4 * (i % 2)) & 0xF
                if dtype[0] == 'i' and val_4b > 7:
                    val_4b = val_4b - 16
                tensor_data_str += f'{val_4b}'
            else:
                tensor_data_str += f'{tensor_data[i]}'
            counter[rank - 1] += 1
            for i in range(rank - 1, 0, -1):
                if counter[i] >= shape[i]:
                    counter[i] = 0
                    counter[i - 1] += 1
                    open_brackets -= 1
                    tensor_data_str += ']'

        # Close all brackets that were opened but not closed yet
        while open_brackets > 0:
            tensor_data_str += ']'
            open_brackets -= 1

        me_node.attrs.append(graph_builder.KeyValue(key='__value', value=tensor_data_str))

    def add_output_tensor_info(self,
                               me_node: graph_builder.GraphNode,
                               tensor_id: int,
                               output_id: int = 0) -> None:
        """Add the output metadata of the node."""
        tensor = self.model.subgraphs[0].tensors[tensor_id]
        # tensor_shape = 'type[shape]' (e.g. 'float32[1,2,3,4]')
        tensor_shape = self.dict_tensor_type_to_string[tensor.type].lower()
        tensor_shape += f'{tensor.shape.tolist()}'
        tensor_name = tensor.name.decode('utf-8')

        metadata = graph_builder.MetadataItem(
            id=f'{output_id}',
            attrs=[
                graph_builder.KeyValue(key='tensor_shape', value=tensor_shape),
                graph_builder.KeyValue(key='tensor_index', value=f'{tensor_id}'),
                graph_builder.KeyValue(key='tensor_name', value=tensor_name)
            ],
        )

        me_node.outputsMetadata.append(metadata)

    def build_graph(self, me_graph: graph_builder.Graph) -> None:
        """Build the graph using the model."""

        sub_graph = self.model.subgraphs[0]

        # Create Input nodes
        node_id = len(sub_graph.operators)
        me_node = graph_builder.GraphNode(id=f'{node_id}',
                                          label="GraphInputs",
                                          namespace="GraphInputs")
        me_graph.nodes.append(me_node)

        for i, tensor_id in enumerate(sub_graph.inputs):
            # Map source and output tensors of GraphInputs
            self.set_source_of(tensor_id=tensor_id, source_id=node_id, output_id=i)
            # Add output metadata to the input node
            self.add_output_tensor_info(me_node=me_node, tensor_id=tensor_id, output_id=i)

        # Map source and output tensors of operators
        for op_id, op in enumerate(sub_graph.operators):
            for i, tensor_id in enumerate(op.outputs):
                self.set_source_of(tensor_id=tensor_id, source_id=op_id, output_id=i)

        # Create pseudo const node for orphan tensors (= const tensors)
        for tensor_id, tensor in enumerate(sub_graph.tensors):
            if (self.get_source_of(tensor_id)) is None:
                node_id += 1
                me_node = graph_builder.GraphNode(id=f'{node_id}',
                                                  label='pseudo_const',
                                                  namespace=tensor.name.decode('utf-8'))
                me_graph.nodes.append(me_node)
                # Map source and output tensors of const tensor
                self.set_source_of(tensor_id=tensor_id, source_id=node_id, output_id=0)
                # Add output metadata to the pseudo const node
                self.add_output_tensor_info(me_node=me_node, tensor_id=tensor_id)
                self.add_tensor_value_attribute(me_node=me_node, tensor_id=tensor_id)

        # Create operator nodes
        for idx, op in enumerate(sub_graph.operators):
            name = self.get_opcode_name(op.opcodeIndex)
            # Construct namespace following output tensor's name
            output_tensor_id = op.outputs[0]
            output_tensor = sub_graph.tensors[output_tensor_id]
            ns = output_tensor.name.decode("utf-8")
            if '/' in ns:
                # Let's take maximum 2 depths of the tensor name
                # '/A/B/C/D' becomes 'A/B'
                ns = '/'.join(ns.strip('/').split('/')[:2])
            me_node = graph_builder.GraphNode(id=f'{idx}', label=name, namespace=ns)
            me_graph.nodes.append(me_node)
            # Add output metadata to the operator node
            self.add_output_tensor_info(me_node=me_node, tensor_id=output_tensor_id)
            # Add operator attributes
            if '__dict__' in dir(op.builtinOptions):
                for k, v in op.builtinOptions.__dict__.items():
                    me_node.attrs.append(graph_builder.KeyValue(key=k, value=f'{v}'))
            # Connect edges from inputs to this operator node
            for i, tensor_id in enumerate(op.inputs):
                if tensor_id < 0:
                    continue
                self.add_incoming_edge(me_node=me_node, tensor_id=tensor_id, input_id=i)

        # Create Output nodes
        node_id += 1
        me_node = graph_builder.GraphNode(id=f'{node_id}',
                                          label="GraphOutputs",
                                          namespace="GraphOutputs")
        me_graph.nodes.append(me_node)
        # Connect edges from inputs to output node
        for i in sub_graph.outputs:
            self.add_incoming_edge(me_node=me_node, tensor_id=i, input_id=i)

    def convert(self, model_path: str, settings: Dict) -> ModelExplorerGraphs:
        """Convert the model to a set of graphs."""
        self.graph = graph_builder.Graph(id='main')

        if settings.get('const_element_count_limit'):
            self.const_element_count_limit = settings['const_element_count_limit']

        self.load_model(model_path)
        self.build_graph(self.graph)

        return {'graphs': [self.graph]}
