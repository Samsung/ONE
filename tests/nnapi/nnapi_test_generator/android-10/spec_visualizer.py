#!/usr/bin/python3

# Copyright 2019, The Android Open Source Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Spec Visualizer

Visualize python spec file for test generator.

Modified from TFLite graph visualizer -- instead of flatbuffer, takes spec file as input.
(https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/visualize.py)

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import fnmatch
import json
import math
import os
import re
import sys
import traceback

# Stuff from test generator
import test_generator as tg
from test_generator import ActivationConverter
from test_generator import BoolScalar
from test_generator import Configuration
from test_generator import DataTypeConverter
from test_generator import DataLayoutConverter
from test_generator import Example
from test_generator import Float32Scalar
from test_generator import Float32Vector
from test_generator import GetJointStr
from test_generator import IgnoredOutput
from test_generator import Input
from test_generator import Int32Scalar
from test_generator import Int32Vector
from test_generator import Internal
from test_generator import Model
from test_generator import Operand
from test_generator import Output
from test_generator import Parameter
from test_generator import ParameterAsInputConverter
from test_generator import RelaxedModeConverter
from test_generator import SmartOpen

# A CSS description for making the visualizer
_CSS = """
<html>
<head>
<style>
body {font-family: sans-serif; background-color: #ffaa00;}
table {background-color: #eeccaa;}
th {background-color: black; color: white;}
h1 {
  background-color: ffaa00;
  padding:5px;
  color: black;
}

div {
  border-radius: 5px;
  background-color: #ffeecc;
  padding:5px;
  margin:5px;
}

.tooltip {color: blue;}
.tooltip .tooltipcontent  {
    visibility: hidden;
    color: black;
    background-color: yellow;
    padding: 5px;
    border-radius: 4px;
    position: absolute;
    z-index: 1;
}
.tooltip:hover .tooltipcontent {
    visibility: visible;
}

.edges line {
  stroke: #333333;
}

.nodes text {
  color: black;
  pointer-events: none;
  font-family: sans-serif;
  font-size: 11px;
}
</style>

<script src="https://d3js.org/d3.v4.min.js"></script>

</head>
<body>
"""

_D3_HTML_TEMPLATE = """
  <script>
    // Build graph data
    var graph = %s;

    var svg = d3.select("#subgraph_%s");
    var width = svg.attr("width");
    var height = svg.attr("height");
    var color = d3.scaleOrdinal(d3.schemeCategory20);

    var simulation = d3.forceSimulation()
        .force("link", d3.forceLink().id(function(d) {return d.id;}))
        .force("charge", d3.forceManyBody())
        .force("center", d3.forceCenter(0.5 * width, 0.5 * height));


    function buildGraph() {
      var edge = svg.append("g").attr("class", "edges").selectAll("line")
        .data(graph.edges).enter().append("line")
      // Make the node group
      var node = svg.selectAll(".nodes")
        .data(graph.nodes)
        .enter().append("g")
        .attr("class", "nodes")
          .call(d3.drag()
              .on("start", function(d) {
                if(!d3.event.active) simulation.alphaTarget(1.0).restart();
                d.fx = d.x;d.fy = d.y;
              })
              .on("drag", function(d) {
                d.fx = d3.event.x; d.fy = d3.event.y;
              })
              .on("end", function(d) {
                if (!d3.event.active) simulation.alphaTarget(0);
                d.fx = d.fy = null;
              }));
      // Within the group, draw a circle for the node position and text
      // on the side.
      node.append("circle")
          .attr("r", "5px")
          .attr("fill", function(d) { return color(d.group); })
      node.append("text")
          .attr("dx", 8).attr("dy", 5).text(function(d) { return d.name; });
      // Setup force parameters and update position callback
      simulation.nodes(graph.nodes).on("tick", forceSimulationUpdated);
      simulation.force("link").links(graph.edges);

      function forceSimulationUpdated() {
        // Update edges.
        edge.attr("x1", function(d) {return d.source.x;})
            .attr("y1", function(d) {return d.source.y;})
            .attr("x2", function(d) {return d.target.x;})
            .attr("y2", function(d) {return d.target.y;});
        // Update node positions
        node.attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; });
      }
    }
  buildGraph()
</script>
"""

class OpCodeMapper(object):
  """Maps an opcode index to an op name."""

  def __init__(self, data):
    self.code_to_name = {}
    for idx, d in enumerate(data["operator_codes"]):
      self.code_to_name[idx] = d["builtin_code"]

  def __call__(self, x):
    if x not in self.code_to_name:
      s = "<UNKNOWN>"
    else:
      s = self.code_to_name[x]
    return "%s (opcode=%d)" % (s, x)


class DataSizeMapper(object):
  """For buffers, report the number of bytes."""

  def __call__(self, x):
    if x is not None:
      return "%d bytes" % len(x)
    else:
      return "--"


class TensorMapper(object):
  """Maps a list of tensor indices to a tooltip hoverable indicator of more."""

  def __init__(self, subgraph_data):
    self.data = subgraph_data

  def __call__(self, x):
    html = ""
    html += "<span class='tooltip'><span class='tooltipcontent'>"
    for i in x:
      tensor = self.data["operands"][i]
      html += str(i) + " "
      html += tensor["name"] + " "
      html += str(tensor["type"]) + " "
      html += (repr(tensor["dimensions"]) if "dimensions" in tensor else "[]") + "<br>"
    html += "</span>"
    html += repr(x)
    html += "</span>"
    return html

def GenerateGraph(g):
  """Produces the HTML required to have a d3 visualization of the dag."""

#   def TensorName(idx):
#     return "t%d" % idx

  def OpName(idx):
    return "o%d" % idx

  edges = []
  nodes = []
  first = {}
  pixel_mult = 50
  for op_index, op in enumerate(g["operations"]):
    for tensor in op["inputs"]:
      if tensor not in first:
        first[str(tensor)] = (
            op_index * pixel_mult,
            len(first) * pixel_mult - pixel_mult / 2)
      edges.append({
          "source": str(tensor),
          "target": OpName(op_index)
      })
    for tensor in op["outputs"]:
      edges.append({
          "target": str(tensor),
          "source": OpName(op_index)
      })
    nodes.append({
        "id": OpName(op_index),
        "name": op["opcode"],
        "group": 2,
        "x": pixel_mult,
        "y": op_index * pixel_mult
    })
  for tensor_index, tensor in enumerate(g["operands"]):
    initial_y = (
        first[tensor["name"]] if tensor["name"] in first else len(g["operations"]))

    nodes.append({
        "id": tensor["name"],
        "name": "%s (%d)" % (tensor["name"], tensor_index),
        "group": 1,
        "x": 2,
        "y": initial_y
    })
  graph_str = json.dumps({"nodes": nodes, "edges": edges})

  html = _D3_HTML_TEMPLATE % (graph_str, g["name"])
  return html

def GenerateTableHtml(items, keys_to_print, display_index=True):
  """Given a list of object values and keys to print, make an HTML table.

  Args:
    items: Items to print an array of dicts.
    keys_to_print: (key, display_fn). `key` is a key in the object. i.e.
      items[0][key] should exist. display_fn is the mapping function on display.
      i.e. the displayed html cell will have the string returned by
      `mapping_fn(items[0][key])`.
    display_index: add a column which is the index of each row in `items`.
  Returns:
    An html table.
  """
  html = ""
  # Print the list of  items
  html += "<table><tr>\n"
  html += "<tr>\n"
  if display_index:
    html += "<th>index</th>"
  for h, mapper in keys_to_print:
    html += "<th>%s</th>" % h
  html += "</tr>\n"
  for idx, tensor in enumerate(items):
    html += "<tr>\n"
    if display_index:
      html += "<td>%d</td>" % idx
    # print tensor.keys()
    for h, mapper in keys_to_print:
      val = tensor[h] if h in tensor else None
      val = val if mapper is None else mapper(val)
      html += "<td>%s</td>\n" % val

    html += "</tr>\n"
  html += "</table>\n"
  return html


def CreateHtmlFile(g, fd):
  """Given a tflite model in `tflite_input` file, produce html description."""
  html = ""

  # Subgraph local specs on what to display
  html += "<div class='subgraph'>"
  tensor_mapper = lambda l: ", ".join(str(op) for op in l)
  op_keys_to_display = [("opcode", None), ("inputs", tensor_mapper), ("outputs", tensor_mapper)]
  tensor_keys_to_display = [("name", None), ("type", None), ("dimensions", None), ("scale", None),
                            ("zero_point", None), ("lifetime", None)]
  html += "<h2>%s</h2>\n" % g["name"]

  # Configurations.
  html += "<h3>Configurations</h3>\n"
  html += GenerateTableHtml(
        [g["options"]], [(k, None) for k in g["options"].keys()], display_index=False)

  # Inputs and outputs.
  html += "<h3>Inputs/Outputs</h3>\n"
  html += GenerateTableHtml(
        [{
            "inputs": g["inputs"],
            "outputs": g["outputs"]
        }], [("inputs", tensor_mapper), ("outputs", tensor_mapper)],
        display_index=False)

  # Print the operands.
  html += "<h3>Operands</h3>\n"
  html += GenerateTableHtml(g["operands"], tensor_keys_to_display)

  # Print the operations.
  html += "<h3>Operations</h3>\n"
  html += GenerateTableHtml(g["operations"], op_keys_to_display)

  # Visual graph.
  html += "<h3>Visual Graph</h3>\n"
  html += "<svg id='subgraph_%s' width='%d' height='%d'></svg>\n"%(
      g["name"], max(min(len(g["operations"])*100, 1600), 200), len(g["operations"])*100)
  html += GenerateGraph(g)
  html += "</div>"

  fd.write(html)

def InitializeHtml(fd):
  html = ""
  html += _CSS
  html += "<h1>%s</h1>"%(tg.FileNames.specName)
  fd.write(html)

def FinalizeHtml(fd):
  fd.write("</body></html>\n")

def VisualizeModel(example, fd):
    if varName is not None and not fnmatch.fnmatch(str(example.testName), varName):
        print("    Skip variation %s" % example.testName)
        return
    print("    Visualizing variation %s" % example.testName)
    model = example.model
    g = {}
    g["options"] = {"relaxed": str(model.isRelaxed), "useSHM": str(tg.Configuration.useSHM())}
    g["name"] = str(example.testName)
    g["inputs"] = model.GetInputs()
    g["outputs"] = model.GetOutputs()
    g["operands"] = [{
            "name": str(op), "type": op.type.type, "dimensions": op.type.GetDimensionsString(),
            "scale": op.type.scale, "zero_point": op.type.zeroPoint, "lifetime": op.lifetime
        } for op in model.operands]
    g["operations"] = [{
            "inputs": op.ins, "outputs": op.outs, "opcode": op.optype
        } for op in model.operations]
    CreateHtmlFile(g, fd)

# Take a model from command line
def ParseCmdLine():
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", help="the spec file")
    parser.add_argument(
        "-v", "--variation", help="the target variation name/pattern", default=None)
    parser.add_argument(
        "-o", "--out", help="the output html path", default="out.html")
    args = parser.parse_args()
    tg.FileNames.InitializeFileLists(
        args.spec, "-", "-", "-", "-", "-")
    tg.FileNames.NextFile()
    return os.path.abspath(args.spec), args.variation, os.path.abspath(args.out)

if __name__ == '__main__':
    specFile, varName, outFile = ParseCmdLine()
    print("Visualizing from spec: %s" % specFile)
    exec(open(specFile, "r").read())
    with SmartOpen(outFile) as fd:
        InitializeHtml(fd)
        Example.DumpAllExamples(
            DumpModel=None, model_fd=None,
            DumpExample=VisualizeModel, example_fd=fd,
            DumpTest=None, test_fd=None)
        FinalizeHtml(fd)
    print("Output HTML file: %s" % outFile)

