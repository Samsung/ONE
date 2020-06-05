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

# read pb file and dump output of specified operations

# some code from https://leimao.github.io/blog/Save-Load-Inference-From-TF-Frozen-Graph/

import tensorflow as tf
from copy import deepcopy
from tensorflow.python.framework import ops as tf_ops

pb_path = "/home/eric/work/A/test/large_graph.pb"
op_names_path = "/home/eric/work/A/test/op_names.txt" # text file. an op name per line.

prefix = "import/"

input_name = "model_inputs"
input_shape = [1, None]
input_dtype = tf.int32

output_name = "model_outputs"
output_shape = []
output_dtype = tf.int32

op_names = [] # run until session reaches to these op


def read_op_names(file_path):
  f = open(file_path, mode='rt', encoding='utf-8')
  for op_name in f:
    if op_name != "":
      op_names.append(op_name.strip())


def load_pb(pb_path):#, placeholder_map):
  with tf.gfile.FastGFile(pb_path, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    graph = tf.import_graph_def(graph_def, name='')

  tf.import_graph_def(graph_def, name="import")

  ops = tf.compat.v1.get_default_graph().get_operations()
  print("Total number of operations: ", len(ops))


# TODO Save tensor values into hdf5 file
def dump():

  graph = tf.compat.v1.get_default_graph()

  input_tensor = graph.get_tensor_by_name(prefix + input_name + ":0")
  output_tensor = graph.get_tensor_by_name(prefix + output_name + ":0")

  sess = tf.InteractiveSession()

  for op_name in op_names:
    tensor = graph.get_tensor_by_name(prefix + op_name+":0")

    result = tensor.eval(
          feed_dict = {input_tensor: [[100, 200, 300, 0, 0, 0]]})

    print("--------------------------------------------------------------")
    print(op_name)
    print(type(result))
    print(result.shape)     # result could be numpy.int32, numpy.ndarray, etc.

  sess.close()


if __name__ == "__main__":

  read_op_names(op_names_path)
  load_pb(pb_path)
  dump()
