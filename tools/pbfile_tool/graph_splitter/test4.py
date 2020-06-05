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

# change op's input to placeholder

# graph_editor/transform.py was used
import tensorflow as tf
from copy import deepcopy
from tensorflow.python.framework import ops as tf_ops


def copy_op_handler(graph, op, new_inputs):

# trying to copy op and change input

  # Clone the node def:
  node_def_ = deepcopy(op.node_def)

  # Transform name:
  name_ = op.name + "_cloned"
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


def main():
  a = tf.constant(1.0, name="const_1")
  b = tf.constant(2.0, name="const_2")
  add1_ = tf.add(a, b, name="add1")

  p1 = tf.compat.v1.placeholder(tf.float32, shape=[1], name = "p1")
  p2 = tf.compat.v1.placeholder(tf.float32, shape=[1], name = "p2")

  copy_op_handler(tf.compat.v1.get_default_graph(), add1_.op, [p1, p2])

  print(tf.compat.v1.get_default_graph().as_graph_def())

  # print(type(add1_.op)) # tensorflow.python.framework.ops.Operation

main()

'''
node {
  name: "p1"
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
          size: 1
        }
      }
    }
  }
}
node {
  name: "p2"
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
          size: 1
        }
      }
    }
  }
}
node {
  name: "add1_cloned"
  op: "Add"
  input: "p1"
  input: "p2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
'''
