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

import pydot
import math

from circle import Model

from pathlib import Path


# Return the name of the tensor
def _tensor_name(graph, tid):
    return graph.Tensors(tid).Name().decode('utf-8')


# Return double-quoted string
def _quote(string: str):
    return '"' + string + '"'


# Class to build dot graph from qerror_map
class DotBuilder:
    def __init__(self, circle_path: str, dot_path: str, metric: str, colors: str):
        '''
        circle_path: Path to the fp32 circle model (required to build graph)
        dot_path: Path to the saved dot file
        metric: Metric name (ex: MPEIR, MSE)
        colors: List of color slots [{'b': begin, 'e': end, 'c':color}, ..]
        '''
        with open(circle_path, 'rb') as f:
            self._model = Model.Model.GetRootAsModel(f.read())

        if self._model.SubgraphsLength() != 1:
            raise RuntimeError("Only one subgraph is supported")

        self._name = Path(circle_path).name
        self._dot_path = dot_path
        self._metric = metric
        self._colors = colors

    # Return color (RGB) for the given qerror
    def _get_color(self, qerror: float):
        # Find a slot where qerror is in the range of [begin, end]
        for slot in self._colors:
            begin = slot['b']
            end = slot['e']
            if (qerror > begin or math.isclose(
                    qerror, begin)) and (qerror < end or math.isclose(qerror, end)):
                return slot['c']

        # Use the first color if qerror is smaller than the first begin
        if qerror < self._colors[0]['b']:
            return self._colors[0]['c']

        # Use the last color if qerror is larger than the last end
        if qerror > self._colors[-1]['e']:
            return self._colors[-1]['c']

        raise RuntimeError("Color ID not found. QError: " + str(qerror))

    # Generate a pydot.Node object which represents the color table
    def _gen_color_table(self):
        color_table = "< <table>"
        for slot in self._colors:
            begin = slot['b']
            end = slot['e']
            color = slot['c']
            color_table += "<tr> <td bgcolor=\""
            color_table += color
            color_table += "\">"
            color_table += self._metric + ": {:.4f}".format(
                begin) + " ~ " + "{:.4f}".format(end)
            color_table += "</td> </tr>"
        color_table += "</table> >"
        return pydot.Node("color_table", shape='none', label=color_table)

    # Save dot graph to self._dot_path
    def save(self, qerror_map: dict):
        '''
        qerror_map: Dictionary of {op_name (str) -> qerror (float)}
        '''
        # Build graph
        DOT = pydot.Dot(self._name, graph_type="digraph")

        # Add color table
        DOT.add_node(self._gen_color_table())

        # Dictionary from output tensor name to Op name {str -> str}
        # This dict is for handling Ops with multiple output tensors.
        # We use the first output tensor's name as the Op name, following
        # the implementation of luci IR
        output_to_op = dict()

        graph = self._model.Subgraphs(0)

        # Add Input nodes
        for i in range(graph.InputsLength()):
            name = _tensor_name(graph, graph.Inputs(i))
            output_to_op[name] = name
            DOT.add_node(pydot.Node(_quote(name)))

        # Add Output nodes
        for i in range(graph.OutputsLength()):
            name = _tensor_name(graph, graph.Outputs(i))
            output_to_op[name] = name
            DOT.add_node(pydot.Node(_quote(name)))

        # Add Edges
        for i in range(graph.OperatorsLength()):
            op = graph.Operators(i)
            # Name of the first output tensor
            op_name = _tensor_name(graph, op.Outputs(0))
            if op.OutputsLength() == 0:
                print(op_name)
                continue

            if op_name in qerror_map:
                qerror = qerror_map[op_name]
                node = pydot.Node(_quote(op_name),
                                  style="filled",
                                  fillcolor=self._get_color(qerror),
                                  xlabel=self._metric + ": {:.4f}".format(qerror))
            else:
                # qerror_map does not have qerror info for the op. Color gray.
                # When this happen? visq does not collect qerror info of some Ops
                # For example, Reshape Op does not change values, so its qerror
                # info is not collected.
                node = pydot.Node(_quote(op_name), style="filled", fillcolor='gray')

            DOT.add_node(node)

            for output_idx in range(op.OutputsLength()):
                output_name = _tensor_name(graph, op.Outputs(output_idx))
                # Set Op name as the first output tensor name (op_name)
                output_to_op[output_name] = op_name

            for j in range(op.InputsLength()):
                op_input = op.Inputs(j)

                # Optional input case (ex: For TConv with no bias, bias is -1)
                if op_input == -1:
                    continue

                op_input_name = _tensor_name(graph, op_input)
                if op_input_name not in output_to_op:
                    continue

                # Use the saved name to handle multiple outputs
                op_input_name = output_to_op[op_input_name]
                DOT.add_edge(pydot.Edge(_quote(op_input_name), _quote(op_name)))

        DOT.write(self._dot_path)
