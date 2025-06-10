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

import pytest
from model_explorer_circle.main import CircleAdapter


@pytest.fixture(scope="module")
def circle_adapter():
    """Fixture to create CircleAdapter instance"""
    obj = CircleAdapter()
    _ = obj.convert('tests/test.circle', settings={})
    return obj


def test_create_graph(circle_adapter):
    """Test if graph is created"""
    me_graph = circle_adapter.graph
    assert me_graph is not None


def test_input_count(circle_adapter):
    """Test if number of inputs matches"""
    circle_model = circle_adapter.model
    circle_graph = circle_model.subgraphs[0]
    me_graph = circle_adapter.graph
    input_node = me_graph.nodes[0]
    assert input_node.label == 'GraphInputs'
    assert len(input_node.outputsMetadata) == len(circle_graph.inputs)


def test_output_count(circle_adapter):
    """Test if number of outputs matches"""
    circle_model = circle_adapter.model
    circle_graph = circle_model.subgraphs[0]
    me_graph = circle_adapter.graph
    output_node = me_graph.nodes[-1]
    assert output_node.label == 'GraphOutputs'
    assert len(output_node.incomingEdges) == len(circle_graph.outputs)


def test_operator_count(circle_adapter):
    """Test if number of operators matches"""
    circle_model = circle_adapter.model
    circle_graph = circle_model.subgraphs[0]
    me_graph = circle_adapter.graph

    # Count operators in circle model
    nr_operators = len(circle_graph.operators)

    # Count operator nodes in model explorer graph
    nr_opnodes = 0
    for node in me_graph.nodes:
        if node.label in ['GraphInputs', 'GraphOutputs', 'pseudo_const']:
            continue
        assert int(node.id) == nr_opnodes
        nr_opnodes += 1

    assert nr_operators == nr_opnodes


def test_const_tensor_count(circle_adapter):
    """Test if number of constant tensors matches"""
    circle_model = circle_adapter.model
    circle_graph = circle_model.subgraphs[0]
    me_graph = circle_adapter.graph

    # Count constant tensors in circle model
    nr_const_tensors = len(circle_graph.tensors)
    nr_const_tensors -= len(circle_graph.inputs)
    nr_const_tensors -= sum([len(op.outputs) for op in circle_graph.operators])

    # Count constant tensor nodes in model explorer graph
    nr_pseudo_const = 0
    for node in me_graph.nodes:
        if node.label == 'pseudo_const':
            nr_pseudo_const += 1
            assert int(node.id) == (len(circle_graph.operators) + nr_pseudo_const)

    assert nr_const_tensors == nr_pseudo_const


def test_unused_tensor(circle_adapter):
    """Test if unused tensors exists"""
    circle_model = circle_adapter.model
    circle_graph = circle_model.subgraphs[0]
    me_graph = circle_adapter.graph

    # Check if all tensors are connected with source nodes
    for tensor_id, _ in enumerate(circle_graph.tensors):
        # Find its source node with tensor id
        src = circle_adapter.get_source_of(tensor_id)
        assert src is not None
        # Source = 'node_id/output_id' (e.g. '11/0')
        src_id, src_out_id = src.split('/')
        src_node = next((n for n in me_graph.nodes if n.id == src_id), None)
        assert src_node is not None

        # Check if the source node has the tensor as one of its outputs metadata
        output_meta = next((o for o in src_node.outputsMetadata if o.id == src_out_id),
                           None)
        assert output_meta is not None
        # Tensor id is in outputsMetatdata.attrs[1]
        attr_tid = output_meta.attrs[1]
        assert attr_tid.key == 'tensor_index'
        # Is it same as tensor id?
        assert int(attr_tid.value) == tensor_id


def test_orphan_node(circle_adapter):
    """Test if orphan nodes exists"""
    me_graph = circle_adapter.graph

    # Check if all nodes are connected with edges
    for node in me_graph.nodes:
        if node.label not in ['GraphInputs', 'pseudo_const']:
            assert len(node.incomingEdges) > 0
        if node.label != 'GraphOutputs':
            assert len(node.outputsMetadata) > 0
