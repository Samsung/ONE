# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
# Copyright 2020 Samsung Electronics co.Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# choose test graph's input and output

'''
Once a pb is imporetd,
original op with, e.g., "Foo" as its name seems to be copied into "imported/Foo" op.
We will make small test graphs (tgraphs), which contains cloned ops, e.g., "cloned/Foo"
'''

import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops

from copy import deepcopy
import queue
from enum import Enum
import sys
import traceback

cloned_prefix = "cloned/"   # clined op will have name, e.g., "cloned/conv2d"
cf_op_types = { "Enter", "Merge", "LoopCond", "Switch", "Exit", "NextIteration" }

max_node_num = 5 # a test graph will have operations <= max_node_num

model_path = "/home/eric/work/A/test/PBSplitterTest1.pb"

input_names  = ["Hole",]
input_shapes = [[None, 2, 3],]
input_dtypes = [tf.float32,]

output_op_names = ["HoleAxis",]
output_op_shapes = [[],]
output_op_dtypes = [tf.int32,]

# model_path = "/home/eric/work/A/test/large_graph.pb"

# input_names  = ["model_inputs",]
# input_shapes = [[1, None],]
# input_dtypes = [tf.int32,]

# output_op_names = ["model_outputs",]
# output_op_shapes = [[],]
# output_op_dtypes = [tf.int32,]


class TraverseMethod(Enum):
  BFS = 1
  DFS = 2


class TGraph:
  def __init__(self):
    self.input_tensors  = [] # starting  cloned placeholders
    self.output_tensors = [] # ending cloned op
    self.linearized_op_names = [] # cloned op names in this tgraph


'''
creates list of Subgraphs

How it works:

E.g., for a graph,

  #0 = placholder
  #1 = const
  #2 = op1(#0, #1)
  #3 = op2(#2)
  #4 = op3(#3)
  #5 = op4(#4)
  #6 = op5(#2)
  #7 = op6(#6)

Calling bfs(g, op1, 3)
returns [ Subgraph([cloned #0], [cloned op3, cloned op5], [op3, op5]),
          Subgraph([cloned input for op4, cloned input for op6], [cloned op4, cloned op6], [op4, op6]))

- assumption: graph is DAG
- note: conflow ops break DAG

'''

def get_op_name(name):
  name_comp = name.split(":")
  if len(name_comp) == 1:
    return name   # name is considered to be an op name

  # when name seems to be a tensor name
  try:
    int(name_comp[1]) # check if second part is int. when not, exception will happen
  except:
    return None  # not tensor name. error

  return name_comp[0]


def is_tensor_name(name):
  name_comp = name.split(":")
  if len(name_comp) != 2:
    return False

  # when name seems to be a tensor name
  try:
    int(name_comp[1]) # check if second part is int. when not, exception will happen
  except:
    return False  # not tensor name. error

  return True


def get_original_op(graph, cloned_tensor_or_op_name):
  cloned_op_name = get_op_name(cloned_tensor_or_op_name)

  assert(cloned_op_name.startswith(cloned_prefix))

  prefix_len = len(cloned_prefix)
  original_name = cloned_op_name[prefix_len:]
  original = graph.get_operation_by_name(original_name)
  assert(original)

  return original


def get_original_tensor(graph, cloned_name):

  assert(is_tensor_name(cloned_name))

  assert(cloned_name.startswith(cloned_prefix))

  prefix_len = len(cloned_prefix)
  original_name = cloned_name[prefix_len:]
  original = graph.get_tensor_by_name(original_name)
  assert(original)

  return original


def get_cloned_name(original_name):
  assert(not original_name.startswith(cloned_prefix))
  return cloned_prefix + original_name


class TensorList(list):
  def append(self, tensor):
    assert(is_tensor_name(tensor.name))
    super().append(tensor)


def duplicate_op(graph, op, new_inputs):

  # trying to copy op and change input

  # Clone the node def:
  node_def_ = deepcopy(op.node_def)

  # Transform name:
  name_ = cloned_prefix + op.name
  node_def_.name = name_

  # Copy the other inputs needed for initialization
  output_types_ = op._output_types[:]    # list of int

  input_types_ = op._input_types[:] # list of 'tensorflow.python.framework.dtypes.DType'

  # Make a copy of the op_def too.
  # Its unique to every _type_ of Operation.
  op_def_ = deepcopy(op.op_def)

  # Initialize a new Operation instance
  try:
    op_ = tf_ops.Operation(node_def_, graph, new_inputs, output_types_,
                         [], input_types_, None, op_def_)
    return op_

  except Exception as ex:
    print(tf.get_default_graph().as_graph_def())
    print("============= op", op.name)
    print("============= new_inputs", new_inputs)
    traceback.print_tb(ex.__traceback__)



class Splitter:

  def __init__(self, max_ops_in_tgraph):
    self.max_ops_in_tgraph = max_ops_in_tgraph
    self.graph = None
    self.use_map = {}
    self.visited_op_set = set()
    self.op_to_visit_q = queue.Queue()


  def clone(self, original_op, is_root, tgraph):

    def add_new_cloned_placeholder(original_tensor, cloned_op_name, new_inputs):
      temp = tf.compat.v1.placeholder(original_tensor.dtype,
                shape = original_tensor.shape,
                name = cloned_op_name)
      new_inputs.append(temp)

    inputs = original_op.inputs

    new_inputs = TensorList()

    # prepare input
    for input in inputs:

      original_input_tensor = self.graph.get_tensor_by_name(input.name)
      original_input_op = self.graph.get_operation_by_name(get_op_name(input.name))

      if original_input_op.type == "Const":
        const_tensor = self.graph.get_tensor_by_name(original_input_op.name + ":0")
        new_inputs.append(const_tensor)
        continue   # let's use this

      # processing, e.g., Conv2D(input, Identity(const)), ..) -> remove Indenty op
      if original_input_op.type == "Identity":
        # is Identity's input is const?
        identity_op = original_input_op
        maybe_const_tensor = identity_op.inputs[0]
        if maybe_const_tensor.op.type == "Const":
          new_inputs.append(maybe_const_tensor)
          continue   # let's use this

      cloned_input_op_name = get_cloned_name(get_op_name(input.name))
      cloned_input_tensor_name = get_cloned_name(input.name)

      # creating inputs
      # if original input is placeholder, name will be cloned_prefix + original_name
      # if original input is not placeholder, name will be cloned_prefix + original_name + "/placeholder"

      # root needs placeholders for all inputs other than const
      if is_root:
        try:
          # input could be some op's output, placeholder (const was handled above)
          root_input_op = self.graph.get_operation_by_name(cloned_input_op_name)

          if root_input_op.type == "Placeholder":
            placeholder = self.graph.get_tensor_by_name(cloned_input_tensor_name)
            new_inputs.append(root_input_op)
          else:
            add_new_cloned_placeholder(original_input_tensor,
              cloned_input_op_name + "/placeholder",  # when previous op exists, create a new placeholder
              new_inputs)                   # TODO Shape is unknown this time. We should add

        except KeyError: # get_operation_by_name fails
            name = ""
            if original_input_tensor.op.type == "Placeholder":
              name = cloned_input_op_name
            else:
              name = cloned_input_op_name + "/placeholder"
            add_new_cloned_placeholder(original_input_tensor, name, new_inputs)
      else:
        try:
          cloned_input_tensor = self.graph.get_tensor_by_name(cloned_input_tensor_name)
          new_inputs.append(cloned_input_tensor)
        except:
          name = ""
          if original_input_tensor.op.type == "Placeholder":
            name = cloned_input_op_name
          else:
            name = cloned_input_op_name + "/placeholder"
          # such placeholder name does not exist. create placeholder
          add_new_cloned_placeholder(original_input_tensor, name, new_inputs)

    return duplicate_op(self.graph, original_op, new_inputs)


  '''
  insert ops looking at tgraph outputs
  '''
  def prepare_next_search(self, tgraph):

    for tgraph_output_tensor in tgraph.output_tensors:
      output_op = get_original_op(self.graph, tgraph_output_tensor.name)

      if (tgraph_output_tensor not in self.use_map):
        break  # leaf op

      for use_op in self.use_map[tgraph_output_tensor]:
        if use_op in self.visited_op_set:
          continue

        if use_op.type in cf_op_types:
          continue

        # print("1. adding op into q", use_op.name)
        self.op_to_visit_q.put(use_op)
        self.visited_op_set.add(use_op)


  def get_tgraph(self):
    if self.op_to_visit_q.empty():
      return None # end

    tgraph = TGraph()

    is_root = True

    while len(tgraph.linearized_op_names) < self.max_ops_in_tgraph:
      if self.op_to_visit_q.empty():
        # print("Q is empty. End of creating tgraph")
        break

      op = self.op_to_visit_q.get()  # root op

      # when meeting "Enter", start of control flow
      # when meeting "LoopCond", end of control flow condition
      # when meeting "NextIteration", end of while body?
      if op.type in cf_op_types: # let's skip
        continue

      # print("Processing tgraph for", op.name)

      new_clone = self.clone(op, is_root, tgraph)
      tgraph.linearized_op_names.append(new_clone.name)

      is_root = False

      for output_tensor in op.outputs:
        output_op_name = get_op_name(output_tensor.name)
        output_op = self.graph.get_operation_by_name(output_op_name)

        if output_tensor not in self.use_map: # leaf tensor
          continue

        next_ops = self.use_map[output_tensor]
        for next_op in next_ops:
          if next_op not in self.visited_op_set:
            if next_op.type not in cf_op_types:
            # print("2. adding op into q", next_op.name)
              self.visited_op_set.add(next_op)
              self.op_to_visit_q.put(next_op)

    # construct input
    for cloned_name in tgraph.linearized_op_names:
      cloned_op = self.graph.get_operation_by_name(cloned_name)
      for cloned_input_tensor in cloned_op.inputs:
        cloned_input_op_name = get_op_name(cloned_input_tensor.name)
        cloned_input_op = self.graph.get_operation_by_name(cloned_input_op_name)
        if cloned_input_op.type == "Placeholder":
          if cloned_input_tensor not in tgraph.input_tensors:
            tgraph.input_tensors.append(cloned_input_tensor)

    # construct output
    output_candidate_op_names = tgraph.linearized_op_names[:]  # copy

    for cloned_op_name in tgraph.linearized_op_names:
      cloned_op = self.graph.get_operation_by_name(cloned_op_name)
      for cloned_input_tensor in cloned_op.inputs:
        try:
          op_name = get_op_name(cloned_input_tensor.name)
          output_candidate_op_names.remove(op_name)
        except:
          pass # in case of placeholders and const, they are not in linearized_op_names

    for output_op_name in output_candidate_op_names:
      output_op = self.graph.get_operation_by_name(output_op_name)
      for output_tensor in output_op.outputs:
        if output_tensor not in tgraph.output_tensors:
          tgraph.output_tensors.append(output_tensor)

    return tgraph


  def _build_tgraphs(self):
    tgraph_list = []
    while True:
      tgraph = self.get_tgraph()

      if tgraph == None or len(tgraph.output_tensors) == 0:
        break

      print("# tgraph found: ")
      print("\t", len(tgraph.input_tensors),"inputs:  ", [tensor.name for tensor in tgraph.input_tensors])
      print("\t", len(tgraph.output_tensors), "outputs:  ", [tensor.name for tensor in tgraph.output_tensors])
      print("\tlinearized: ", tgraph.linearized_op_names)

      tgraph_list.append(tgraph)

      self.prepare_next_search(tgraph)


    return tgraph_list


  def _append_tgraphs(self, tgraph_list, new_tgraphs):
    for new_tgraph in new_tgraphs:
      tgraph_list.append(new_tgraph)


  def split(self, input_names):

    tgraph_list = []
    nodes_in_q = set()

    #debug
    # print("")
    # for key in self.use_map:
    #   print("------->", key, ":", self.use_map[key])

    # Part 1. graphs from input placeholders of graph
    # initial queuing
    for input_name in input_names:
      input_tensor = self.graph.get_tensor_by_name(input_name)
      starting_ops = self.use_map[input_tensor]
      for starting_op in starting_ops:
        # print("3. adding op into q", starting_op.name)
        if starting_op.type not in cf_op_types:
          self.op_to_visit_q.put(starting_op)
          self.visited_op_set.add(starting_op)

    tgraph_list = self._build_tgraphs()

    # Part 2. find cf(control flow) ops. skip them and find the next non-cf op
    ops = self.graph.get_operations()
    cf_ops = [op for op in ops if (not op.name.startswith(cloned_prefix)) and (op.type in cf_op_types) ]

    # if the next op after control flow op is not visited non-cf op, prepare to process
    for cf_op in cf_ops:
      if cf_op in self.visited_op_set:
        continue

      self.visited_op_set.add(cf_op) # mark it visited

      for cf_op_output_tensor in cf_op.outputs:
        if cf_op_output_tensor in self.use_map:
          next_ops = self.use_map[cf_op_output_tensor]

          for next_op in next_ops:
            if (next_op.type not in cf_op_types) and (next_op not in self.visited_op_set):
              self.op_to_visit_q.put(next_op)
              self.visited_op_set.add(next_op)

              new_tgraphs = self._build_tgraphs()
              self._append_tgraphs(tgraph_list, new_tgraphs)

    return tgraph_list

  '''
  returns a map of { tensor_name, [op1, op2, .., opn] } where tensor_name is input of op1, op2, etc.
  use_map is based on original ops
  '''
  def construct_input_use_map(self):
    self.use_map = {}

    ops = self.graph.get_operations()

    for op in ops:

      input_tensors = op.inputs

      for input_tensor in input_tensors:
        if input_tensor not in self.use_map:
          self.use_map[input_tensor] = [op]
        else:
          self.use_map[input_tensor].append(op)


  def load_model(self, model_path):
    if model_path.endswith(".pb"):

      with tf.gfile.FastGFile(model_path, 'rb') as f:
          graph_def = tf.compat.v1.GraphDef()
          graph_def.ParseFromString(f.read())
          graph = tf.import_graph_def(graph_def, name='')

    elif model_path.endswith(".pbtxt"):

      from google.protobuf import text_format as pbtf
      from tensorflow.core.framework import graph_pb2 as gpb

      graph_def = gpb.GraphDef()
      with open(model_path, 'r') as fh:
        graph_str = fh.read()
      pbtf.Parse(graph_str, graph_def)
      graph = tf.import_graph_def(graph_def, name='')

    def check_naming():
      ops = tf.compat.v1.get_default_graph().get_operations()
      for op in ops:
        assert(not op.name.startswith(cloned_prefix))

    check_naming()

    self.graph = tf.compat.v1.get_default_graph()

    # print("---------- original graph --------------\n")
    # print(self.graph.as_graph_def())


def main():
  # main
  splitter = Splitter()

  splitter.load_model(model_path)

  splitter.construct_input_use_map()

  tgraph_list = splitter.split(model_path, input_names, max_node_num, tgraph_list)



# unit test

import unittest

class SimpleGraphTest(unittest.TestCase):

  def setUp(self):
    self.model_path = "/home/eric/work/A/test/PBSplitterTest1.pb"
    self.input_list = ["Hole:0", "HoleAxis:0"]
    self.output_op_names = ["Relu6"]
    self.splitter = None

  def tearDown(self):
    tf.compat.v1.reset_default_graph()
    return super().tearDown()


  def _assert_tgraph(self, tgraph, num_inputs, num_outputs, input_name_list, output_name_list):
    assert(len(tgraph.input_tensors) == num_inputs)
    assert(len(tgraph.output_tensors) == num_outputs)
    assert(len([n for n in input_name_list if self.splitter.graph.get_tensor_by_name(n) in tgraph.input_tensors]) == num_inputs)
    assert(len([n for n in output_name_list if self.splitter.graph.get_tensor_by_name(n) in tgraph.output_tensors]) == num_outputs)


  def test_load(self):
    self.splitter = Splitter(4)
    self.splitter.load_model(self.model_path)
    self.assertEqual(len(self.splitter.graph.get_operations()), 10)


  def test_construct_use_map(self):
    self.splitter = Splitter(4)
    self.splitter.load_model(self.model_path)
    self.splitter.construct_input_use_map()

    self.assertEqual(len(self.splitter.use_map), 9)

    shape = self.splitter.graph.get_operation_by_name("Shape")
    shape_output = shape.outputs[0]
    reshape = self.splitter.graph.get_operation_by_name("Reshape")
    use_list = self.splitter.use_map[shape_output]
    self.assertEqual(len(use_list), 1)
    self.assertEqual(reshape in use_list, True)


  def test_get_original(self):
    self.splitter = Splitter(4)
    self.splitter.load_model(self.model_path)
    original_shape = self.splitter.graph.get_operation_by_name("Shape")

    original = get_original_op(self.splitter.graph, cloned_prefix + "Shape")
    self.assertEqual(original_shape, original)


  def test_split_PBSplitterTest1_pb(self):
    self.model_path = "/home/eric/work/A/test/PBSplitterTest1.pb"
    self.input_list = ["Hole:0", "HoleAxis:0"]
    self.output_op_names = ["Relu6"]

    self.splitter = Splitter(4)
    self.splitter.load_model(self.model_path)

    self.splitter.construct_input_use_map()
    tgraph_list = self.splitter.split(self.input_list)

    assert(len(tgraph_list) == 2)

    graph = self.splitter.graph

    tgraph = tgraph_list[0]
    assert(len(tgraph.input_tensors) == 2)
    assert(len(tgraph.output_tensors) == 1)
    assert(graph.get_tensor_by_name("cloned/Hole:0") in tgraph.input_tensors)
    assert(graph.get_tensor_by_name("cloned/HoleAxis:0") in tgraph.input_tensors)
    assert(graph.get_tensor_by_name("cloned/ExpandDims2:0") in tgraph.output_tensors)

    tgraph = tgraph_list[1]
    assert(len(tgraph.input_tensors) == 1)
    assert(len(tgraph.output_tensors) == 1)
    assert(graph.get_tensor_by_name("cloned/ExpandDims2/placeholder:0") in tgraph.input_tensors)
    assert(graph.get_tensor_by_name("cloned/Relu6:0") in tgraph.output_tensors)


  def test_split_mobilenet_pb(self):
    self.model_path = "/home/eric/src/models/mobilenet/mobilenet_v1_0.25_128_frozen.pb"
    self.input_list = ["input:0"]
    # self.output_op_names = ["Relu6"]

    self.splitter = Splitter(50)
    self.splitter.load_model(self.model_path)

    self.splitter.construct_input_use_map()
    tgraph_list = self.splitter.split(self.input_list)

    assert(len(tgraph_list) == 2)

    graph = self.splitter.graph

    self._assert_tgraph(tgraph_list[0], 1, 5,
      ["cloned/input:0"],
      ["cloned/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNorm:0",
        "cloned/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNorm:1",
        "cloned/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNorm:2",
        "cloned/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNorm:3",
        "cloned/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNorm:4",])
    assert(len(tgraph_list[0].linearized_op_names) == 50)

    self._assert_tgraph(tgraph_list[1], 1, 1,
      ["cloned/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNorm/placeholder:0"],
      ["cloned/MobilenetV1/Predictions/Reshape_1:0"])
    assert(len(tgraph_list[1].linearized_op_names) == 40) # manually counted


  def test_split_while_pb(self):
    self.model_path = "/home/eric/work/A/test/while.pb"
    self.input_list = ["Hole1:0", "Hole2:0"]

    self.splitter = Splitter(5)
    self.splitter.load_model(self.model_path)

    self.splitter.construct_input_use_map()
    tgraph_list = self.splitter.split(self.input_list)

    assert(len(tgraph_list) == 4)

    graph = self.splitter.graph

    self._assert_tgraph(tgraph_list[0], 1, 1,
      ['cloned/While/Merge/placeholder:0'],
      ['cloned/While/Less:0'])

    self._assert_tgraph(tgraph_list[1], 1, 1,
      ["cloned/While/Switch/placeholder:0"],
      ["cloned/While/Add:0"])

    self._assert_tgraph(tgraph_list[2], 1, 1,
      ["cloned/While/Switch_1/placeholder:0"],
      ["cloned/While/Tanh:0"])

    self._assert_tgraph(tgraph_list[3], 1, 1,
      ["cloned/While/Exit_1/placeholder:0"],
      ["cloned/Relu:0"])


  def test_split_if_pb(self):
    self.model_path = "/home/eric/work/1/test_model/if_ending_with_relu.pbtxt"
    self.input_list = ["input:0"]

    self.splitter = Splitter(5)
    self.splitter.load_model(self.model_path)

    self.splitter.construct_input_use_map()
    tgraph_list = self.splitter.split(self.input_list)

    # two tgraph contains one Identity only
    assert(len(tgraph_list) == 6)

    graph = self.splitter.graph

    self._assert_tgraph(tgraph_list[0], 1, 1,
      ['cloned/input:0'],
      ['cloned/Cond/pred_id:0'])

    self._assert_tgraph(tgraph_list[1], 1, 1,
      ['cloned/Cond/Switch/placeholder:0'],
      ['cloned/Cond/switch_f:0'])

    self._assert_tgraph(tgraph_list[2], 1, 1,
      ['cloned/Cond/Switch/placeholder_1:0'],
      ['cloned/Cond/switch_t:0'])

    self._assert_tgraph(tgraph_list[3], 1, 1,
      ['cloned/Cond/Mul/Switch/placeholder:0'],
      ['cloned/Cond/Mul:0'])

    self._assert_tgraph(tgraph_list[4], 1, 1,
      ['cloned/Cond/Add/Switch/placeholder:0'],
      ['cloned/Cond/Add:0'])

    self._assert_tgraph(tgraph_list[5], 1, 1,
      ['cloned/Cond/Merge/placeholder:0'],
      ['cloned/Relu:0'])


  def test_split_large_pb(self):
    self.model_path = "/home/eric/work/A/test/large_graph.pb"
    self.input_list = ["model_inputs:0"]

    self.splitter = Splitter(10)
    self.splitter.load_model(self.model_path)

    self.splitter.construct_input_use_map()
    tgraph_list = self.splitter.split(self.input_list)

    print(len(tgraph_list))


if __name__ == "__main__":
  #main()
  unittest.main()

