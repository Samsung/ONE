#!/usr/bin/python

# Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

import os
import sys
import platform
import tensorflow as tf
import argparse

import base_freezer as base
import model_freezer_util as util


# see MUL_gen.py for details usage and sample
class GenFloor(base.BaseFreezer):
    def __init__(self, path):
        super(GenFloor, self).__init__(path)

    def getOutputDirectory(self):
        return os.path.join(self.root_output_path,
                            'floor')  # the root path of generated files

    def getTestCases(self):
        return {"floor_4d_4d": [base.Tensor([1, 2, 2, 1]), base.Tensor([1, 2, 2, 1])]}

    def buildModel(self, sess, test_case_tensor, tc_name):

        input_list = []

        x_tensor = self.createTFInput(test_case_tensor[0], input_list)

        output_node = tf.floor(x_tensor, name=tc_name)

        # ------ modify UNTIL here for your model -------#

        # Note if don't have any CONST value, creating checkpoint file fails.
        # The next lines insert such (CONST) to prevent such error.
        # So, Graph.pb/pbtxt contains this garbage info,
        # but this garbage info will be removed in Graph_frozen.pb/pbtxt
        garbage = tf.get_variable("garbage", [1],
                                  dtype=tf.float32,
                                  initializer=tf.zeros_initializer())
        init_op = tf.global_variables_initializer()
        garbage_value = [0]
        sess.run(tf.assign(garbage, garbage_value))

        sess.run(init_op)

        # ------ modify appropriate return value  -------#

        # returning (input_node_list, output_node_list)
        return (input_list, [output_node])


class GenPad(base.BaseFreezer):
    def __init__(self, path):
        super(GenPad, self).__init__(path)

    def getOutputDirectory(self):
        return os.path.join(self.root_output_path,
                            'pad')  # the root path of generated files

    def getTestCases(self):
        return {
            "pad_4d_2d": [
                base.Tensor([1, 2, 2, 1]),
                base.Tensor([4, 2], dtype=tf.int32, const_val=[0, 0, 1, 1, 1, 1, 0, 0])
            ]
        }

    def buildModel(self, sess, test_case_tensor, tc_name):

        input_list = []

        input_tensor = self.createTFInput(test_case_tensor[0], input_list)
        pad_tensor = self.createTFInput(test_case_tensor[1], input_list)

        output_node = tf.pad(input_tensor, pad_tensor, name=tc_name)

        # ------ modify UNTIL here for your model -------#

        # Note if don't have any CONST value, creating checkpoint file fails.
        # The next lines insert such (CONST) to prevent such error.
        # So, Graph.pb/pbtxt contains this garbage info,
        # but this garbage info will be removed in Graph_frozen.pb/pbtxt
        garbage = tf.get_variable("garbage", [1],
                                  dtype=tf.float32,
                                  initializer=tf.zeros_initializer())
        init_op = tf.global_variables_initializer()
        garbage_value = [0]
        sess.run(tf.assign(garbage, garbage_value))

        sess.run(init_op)

        # ------ modify appropriate return value  -------#

        # returning (input_node_list, output_node_list)
        return (input_list, [output_node])


class GenSqueeze(base.BaseFreezer):
    def __init__(self, path):
        super(GenSqueeze, self).__init__(path)

    def getOutputDirectory(self):
        return os.path.join(self.root_output_path,
                            'squeeze')  # the root path of generated files

    def getTestCases(self):
        return {"squeeze_3d": [base.Tensor([1, 5, 1])]}

    def buildModel(self, sess, test_case_tensor, tc_name):

        input_list = []

        input_tensor = self.createTFInput(test_case_tensor[0], input_list)

        output_node = tf.squeeze(input_tensor, [2], name=tc_name)

        # ------ modify UNTIL here for your model -------#

        # Note if don't have any CONST value, creating checkpoint file fails.
        # The next lines insert such (CONST) to prevent such error.
        # So, Graph.pb/pbtxt contains this garbage info,
        # but this garbage info will be removed in Graph_frozen.pb/pbtxt
        garbage = tf.get_variable("garbage", [1],
                                  dtype=tf.float32,
                                  initializer=tf.zeros_initializer())
        init_op = tf.global_variables_initializer()
        garbage_value = [0]
        sess.run(tf.assign(garbage, garbage_value))

        sess.run(init_op)

        # ------ modify appropriate return value  -------#

        # returning (input_node_list, output_node_list)
        return (input_list, [output_node])


class GenTranspose(base.BaseFreezer):
    def __init__(self, path):
        super(GenTranspose, self).__init__(path)

    def getOutputDirectory(self):
        return os.path.join(self.root_output_path,
                            'transpose')  # the root path of generated files

    def getTestCases(self):
        return {"transpose_4d": [base.Tensor([1, 2, 2, 1])]}

    def buildModel(self, sess, test_case_tensor, tc_name):

        input_list = []

        input_tensor = self.createTFInput(test_case_tensor[0], input_list)

        output_node = tf.transpose(input_tensor, [0, 2, 1, 3], name=tc_name)

        # ------ modify UNTIL here for your model -------#

        # Note if don't have any CONST value, creating checkpoint file fails.
        # The next lines insert such (CONST) to prevent such error.
        # So, Graph.pb/pbtxt contains this garbage info,
        # but this garbage info will be removed in Graph_frozen.pb/pbtxt
        garbage = tf.get_variable("garbage", [1],
                                  dtype=tf.float32,
                                  initializer=tf.zeros_initializer())
        init_op = tf.global_variables_initializer()
        garbage_value = [0]
        sess.run(tf.assign(garbage, garbage_value))

        sess.run(init_op)

        # ------ modify appropriate return value  -------#

        # returning (input_node_list, output_node_list)
        return (input_list, [output_node])


# How to run
# $ chmod +x tools/tensorflow_model_freezer/sample/name_of_this_file.py
# $ PYTHONPATH=$PYTHONPATH:./tools/tensorflow_model_freezer/ \
#      tools/tensorflow_model_freezer/sample/name_of_this_file.py \
#      ~/temp  # directory where model files are saved
# --------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Converted Tensorflow model in python to frozen model.')
    parser.add_argument(
        "out_dir",
        help=
        "directory where generated pb, pbtxt, checkpoint and Tensorboard log files are stored."
    )

    args = parser.parse_args()
    root_output_path = args.out_dir

    GenFloor(root_output_path).createSaveFreezeModel()
    GenPad(root_output_path).createSaveFreezeModel()
    GenSqueeze(root_output_path).createSaveFreezeModel()
    GenTranspose(root_output_path).createSaveFreezeModel()
