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


# Class to build dot graph from qerror_map
class DotBuilder:
    def __init__(self, circle_path: str, dot_path: str, metric: str, colors: str):
        '''
        circle_path: Path to the fp32 circle model (required to build graph)
        dot_path: Path to the saved dot file
        metric: Metric name (ex: MPEIR, MSE)
        colors: list ['b': begin, 'e': end, 'c':color]
        '''
        with open(circle_path, 'rb') as f:
            self._model = Model.Model.GetRootAsModel(f.read(), 0)
        self._name = Path(circle_path).name
        self._dot_path = dot_path
        self._metric = metric
        self._colors = colors

    def _get_color(self, qerror):
        for slot in self._colors:
            begin = slot['b']
            end = slot['e']
            if (qerror > begin or math.isclose(
                    qerror, begin)) and (qerror < end or math.isclose(qerror, end)):
                return slot['c']

        if qerror < self._colors[0]['b']:
            return self._colors[0]['c']
        if qerror > self._colors[-1]['e']:
            return self._colors[-1]['c']

        raise RuntimeError("Color ID not found. QError: " + str(qerror))

    def save(self, qerror_map: dict):
        '''
        qerror_map: dict of {op_name (str) -> qerror (float)}
        '''
        # Build graph
        DOT = pydot.Dot(self._name, graph_type="digraph")

        graph = self._model.Subgraphs(0)

        # Build color table
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
        DOT.add_node(pydot.Node("color_table", shape='none', label=color_table))

        # Dictionary from output tensor name to Op name {str -> str}
        # This dict is for handling Ops with multiple output tensors.
        # We use the first output tensor's name as the Op name, following
        # the implementation of luci IR
        output_to_op = dict()

        # Add Input nodes
        for i in range(graph.InputsLength()):
            input_tensor = graph.Tensors(graph.Inputs(i))
            name = input_tensor.Name().decode('utf-8')
            output_to_op[name] = name
            DOT.add_node(pydot.Node(name))

        # Add Output nodes
        for i in range(graph.OutputsLength()):
            output_tensor = graph.Tensors(graph.Outputs(i))
            name = output_tensor.Name().decode('utf-8')
            output_to_op[name] = name
            DOT.add_node(pydot.Node(name))

        # Add Edges
        for i in range(graph.OperatorsLength()):
            op = graph.Operators(i)
            # Name of the first output tensor
            op_name = graph.Tensors(op.Outputs(0)).Name().decode('utf-8')
            if op.OutputsLength() == 0:
                print(op_name)
                continue

            if op_name in qerror_map:
                qerror = qerror_map[op_name]
                node = pydot.Node(
                    op_name,
                    style="filled",
                    fillcolor=self._get_color(qerror),
                    xlabel=self._metric + ": {:.4f}".format(qerror))
            else:
                # No qerror info. Color gray.
                node = pydot.Node(op_name, style="filled", fillcolor='gray')

            DOT.add_node(node)

            for output_idx in range(op.OutputsLength()):
                output_name = graph.Tensors(op.Outputs(output_idx)).Name().decode('utf-8')
                # Set Op name as the first output tensor name (op_name)
                output_to_op[output_name] = op_name

            for j in range(op.InputsLength()):
                op_input = op.Inputs(j)

                # Optional input case (ex: For TConv with no bias, bias is -1)
                if op_input == -1:
                    continue

                op_input_name = graph.Tensors(op_input).Name().decode('utf-8')
                if op_input_name not in output_to_op:
                    continue

                # Use the saved name to handle multiple outputs
                op_input_name = output_to_op[op_input_name]
                DOT.add_edge(pydot.Edge(op_input_name, op_name))

        DOT.write(self._dot_path)
