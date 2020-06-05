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

# evolved version of test6.py

# given input names and output names,
# this runs tensorflow and save input and output values into hdf5 file

# some code from https://leimao.github.io/blog/Save-Load-Inference-From-TF-Frozen-Graph/

import tensorflow as tf
from copy import deepcopy
from tensorflow.python.framework import ops as tf_ops


def load_pb(pb_path):#, placeholder_map):
  with tf.gfile.FastGFile(pb_path, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    graph = tf.import_graph_def(graph_def, name='')

  tf.import_graph_def(graph_def, name="")

  ops = tf.compat.v1.get_default_graph().get_operations()
  print("Total number of operations: ", len(ops))



import numpy as np
import h5py


'''
  class to read or write numpy data
  the hdf5 format should be aligned with tests/tools/nnpackage_run/src/h5formatter.cc
'''
class H5:

  def __init__(self, filepath, mode):
    assert(mode in ['r', 'w'])
    self.group_name = "value"
    self.f = h5py.File(filepath, mode)
    self.group = self.f.create_group(self.group_name)

  def write(self, index, numpy_value):  #numpy_values is type of numpy.int32 (scalar), numpy.ndarray, ..
    assert(type(index) is int)
    self.group.create_dataset(str(index), shape=numpy_value.shape, data=numpy_value)

  def finish(self):
    self.f.close()

'''
This should be run after creating tf.InteractiveSession()
feed_dict is the {input_tensor1: val, input_tensor2: val, ..} of  the whole model
'''
def save_tf_input_output(input_tensor_name_list, output_tensor_name_list, feed_dict, h5_filename_input, h5_filename_output):
    graph = tf.compat.v1.get_default_graph()

    def run_tf_and_save(output_tensor_name_list, feed_dict, h5_filename):

      h5 = H5(h5_filename, 'w')

      index = 0
      for output_tensor_name in output_tensor_name_list:
        output_tensor = graph.get_tensor_by_name(output_tensor_name)
        result = output_tensor.eval(feed_dict = feed_dict)

        h5.write(index, result)
        index += 1

    # save input tensor info
    run_tf_and_save(input_tensor_name_list, feed_dict, h5_filename_input)

    # save output tensor info
    run_tf_and_save(output_tensor_name_list, feed_dict, h5_filename_output)


if __name__ == "__main__":

  load_pb("/home/eric/work/A/test/PBSplitterTest1.pb")

  graph = tf.compat.v1.get_default_graph()

  hole = graph.get_tensor_by_name("Hole:0")
  hole_val = [[[1, 1, 1], [2, 2, 2]]]

  holeAxis = graph.get_tensor_by_name("HoleAxis:0")
  holeAxis_val = [0]

  feed_dict = {hole: hole_val, holeAxis: holeAxis_val}

  sess = tf.InteractiveSession()

  save_tf_input_output(['Hole:0', 'HoleAxis:0'],
                       ['ExpandDims2:0'],
                       feed_dict,
                       "tgraph_00_input.h5",
                       "tgraph_00_output.h5")

  save_tf_input_output(['ExpandDims2:0'],  # don't forget to remove "cloned/" and "placeholder"
                       ['Relu6:0'],
                       feed_dict,
                       "tgraph_01_input.h5",
                       "tgraph_01_output.h5")
  sess.close()
