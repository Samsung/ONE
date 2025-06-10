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


class Gen(base.BaseFreezer):
    '''
    class to generate tflite file for TOPK
    '''
    def __init__(self, path):
        super(Gen, self).__init__(path)

    def getOutputDirectory(self):
        return os.path.join(self.root_output_path,
                            'topk')  # the root path of generated files

    def getTestCases(self):
        '''
        this returns a hash of test case (= set of input type), for example:
            [1.2, -2.3] : two input, both are scalar. one is 1.2, another is -2.3
            [[5,3], [5,4,3]] : two input, both are shapes. one is [5.3], another is [5,4,3]

        test name (key of hash) is used as
            - prefix of file name to be generated
            - output node name pf graph
        '''
        return {
            "topk_2d": [
                base.Tensor(shape=[2, 3], dtype=tf.float32),
                base.Tensor(shape=[], const_val=2, dtype=tf.int32)
            ],
            "topk_3d": [
                base.Tensor(shape=[2, 3, 4], dtype=tf.float32),
                base.Tensor(shape=[], const_val=2, dtype=tf.int32)
            ],
        }

    def buildModel(self, sess, test_case_tensor, tc_name):
        '''
        please, refer to the comment in MUL_gen.py to see how to rewrite this method
        '''

        input_list = []
        output_list = []

        # ------ modify below for your model FROM here -------#

        x_tensor = self.createTFInput(test_case_tensor[0], input_list)
        y_tensor = self.createTFInput(test_case_tensor[1], input_list)

        # defining output node and input list
        values_op, indices_op = tf.nn.top_k(
            x_tensor,
            y_tensor,  # add your input here
            name=tc_name)  # do not modify name

        output_list.append(values_op)
        output_list.append(indices_op)
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
        return (input_list, output_list)


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

    Gen(root_output_path).createSaveFreezeModel()
