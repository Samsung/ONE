# what happens when all inputs are not replaced to placeholder?

import tensorflow as tf
import copy
import tensorflow.contrib.graph_editor as ge
from copy import deepcopy

a = tf.constant(1.0, name="const_1")
b = tf.constant(2.0, name="const_2")
add1_ = tf.add(a, b, name="add1")
tanh_ = tf.nn.tanh(add1_, name="tanh")
add2_ = tf.add(add1_, tanh_, name="add2")
relu_ = tf.nn.relu(add2_, name="relu")
f = tf.nn.relu6(relu_, name="relu6")

# let's replace tanh_ to placeholder
# but not add1_

print(tf.get_default_graph().as_graph_def())

print("----------------------")

def toPlaceholder(t):
    t_graphdef = t.op.node_def
    placeholder_name = t_graphdef.name + "_to_placeholder_by_one"
    a = tf.compat.v1.placeholder(tf.float32, shape=[1], name = placeholder_name)
    return a.op.outputs[0]

def update_existing(target, updated):
    # illustrate how to use new op
    related_ops = ge.get_backward_walk_ops(target, stop_at_ts=updated.keys(), inclusive=True)
    new_ops, mapping = ge.copy_with_input_replacements(related_ops, updated)
    new_op = mapping._transformed_ops[target.op]
    return new_op.outputs[0]

tanh_to_placeholder = toPlaceholder(tanh_)

output = update_existing(relu_, {tanh_ : tanh_to_placeholder})

print(tf.compat.v1.get_default_graph().as_graph_def())

'''
output contains add1 and two consts

node {
  name: "tanh_to_placeholder_by_one"
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
  name: "const_1_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "const_2_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 2.0
      }
    }
  }
}
node {
  name: "add1_1"
  op: "Add"
  input: "const_1_1"
  input: "const_2_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "add2_1"
  op: "Add"
  input: "add1_1"
  input: "tanh_to_placeholder_by_one"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "relu_1"
  op: "Relu"
  input: "add2_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}

'''