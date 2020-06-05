

import tensorflow as tf
import copy
import tensorflow.contrib.graph_editor as ge
from copy import deepcopy

a = tf.constant(1.0, name="const_1")
b = tf.constant(2.0, name="const_2")
add_ = tf.add(a, b, name="add")
tahn_ = tf.nn.tanh(add_, name="tanh")
relu_ = tf.nn.relu(tahn_, name="relu")
f = tf.nn.relu6(relu_, name="relu6")

print(tf.get_default_graph().as_graph_def())

print("----------------------")

def toPlaceholder(t):
    t_graphdef = t.op.node_def
    placeholder_name = t_graphdef.name + "_to_placeholder_by_one"
    a = tf.compat.v1.placeholder(tf.float32, shape=[1], name = placeholder_name)
    return a.op.outputs[0]

def modify(t):
    # illustrate operation copy&modification
    new_t = deepcopy(t.op.node_def)
    new_t.name = new_t.name+"_but_awesome"
    new_t = tf.Operation(new_t, tf.get_default_graph())
    # we got a tensor, let's return a tensor
    return new_t.outputs[0]

def update_existing(target, updated):
    # illustrate how to use new op
    related_ops = ge.get_backward_walk_ops(target, stop_at_ts=updated.keys(), inclusive=True)
    new_ops, mapping = ge.copy_with_input_replacements(related_ops, updated)
    new_op = mapping._transformed_ops[target.op]
    return new_op.outputs[0]

add_to_placeholder = toPlaceholder(add_)

output = update_existing(relu_, {add_ : add_to_placeholder})

print(tf.compat.v1.get_default_graph().as_graph_def())

'''
output contains

node {
  name: "add_to_placeholder_by_one"
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
  name: "tanh_1"
  op: "Tanh"
  input: "add_to_placeholder_by_one"
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
  input: "tanh_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}

'''