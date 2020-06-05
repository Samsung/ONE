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

# read pb file and change op's input to placeholder

# some code from https://leimao.github.io/blog/Save-Load-Inference-From-TF-Frozen-Graph/

import tensorflow as tf
from copy import deepcopy
from tensorflow.python.framework import ops as tf_ops

pb_path = "/home/eric/work/A/test/PBSplitterTest1.pb"
name_prefix = "one_test/"

def duplicate_op(graph, op, new_inputs):

  # trying to copy op and change input

  # Clone the node def:
  node_def_ = deepcopy(op.node_def)

  # Transform name:
  name_ = name_prefix + op.name
  node_def_.name = name_

  # Copy the other inputs needed for initialization
  output_types_ = op._output_types[:]    # list of int

  input_types_ = op._input_types[:] # list of 'tensorflow.python.framework.dtypes.DType'

  # Make a copy of the op_def too.
  # Its unique to every _type_ of Operation.
  op_def_ = deepcopy(op.op_def)

  # Initialize a new Operation instance
  op_ = tf_ops.Operation(node_def_, graph, new_inputs, output_types_,
                         [], input_types_, None, op_def_)


def load_pb(pb_path):
  with tf.gfile.FastGFile(pb_path, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      graph = tf.import_graph_def(graph_def, name='')

  nodes = [n.name + ' => ' +  n.op for n in graph_def.node if n.op in ('Placeholder')]
  for node in nodes:
      print(node)

  hole = tf.compat.v1.placeholder(tf.float32, shape=[4, 2, 3], name = "Hole")
  holeAxis = tf.compat.v1.placeholder(tf.int32, shape=[], name = "HoleAxis")

  tf.import_graph_def(graph_def, {'Hole': hole, 'HoleAxis': holeAxis})


def main():

  graph_def = load_pb(pb_path)
  #[print(n) for n in graph_def.node if n.op in ('Placeholder')]

  p1 = tf.compat.v1.placeholder(tf.float32, shape=[4, 2, 3], name = name_prefix + "Hole")
  p2 = tf.compat.v1.placeholder(tf.int32, shape=[], name = name_prefix + "HoleAxis")

  graph = tf.compat.v1.get_default_graph()

  expand_dim = graph.get_operation_by_name("ExpandDims2")

  duplicate_op(tf.compat.v1.get_default_graph(), expand_dim, [p1, p2])

  print(tf.compat.v1.get_default_graph().as_graph_def())

main()

'''

...

node {
  name: "one_test/Hole"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 4
        }
        dim {
          size: 2
        }
        dim {
          size: 3
        }
      }
    }
  }
}
node {
  name: "one_test/HoleAxis"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
}
node {
  name: "one_test/ExpandDims2"
  op: "ExpandDims"
  input: "one_test/Hole_new_inpu"
  input: "one_test/HoleAxis"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tdim"
    value {
      type: DT_INT32
    }
  }
}
'''
