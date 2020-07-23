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
    class to generate tflite files for MUL
    '''
    def __init__(self, path):
        super(Gen, self).__init__(path)

    def getOutputDirectory(self):
        return os.path.join(self.root_output_path,
                            'argmin')  # the root path of generated files

    def getTestCases(self):
        '''
        this returns a a hash containg test cases.
        key of return hash is test case name and
        value of return hash is test is a list of input tensor metadata.
        test name (key of hash) is used as
            - prefix of file name to be generated (don't use white space or special characters)
            - output node name pf graph
        '''
        return {"argmin_4d": [base.Tensor([1, 2, 4, 3])]}

    def buildModel(self, sess, test_case_tensor, tc_name):
        '''
        This method is called per test case (defined by getTestCases()).

        keyword argument:
        test_case_tensor -- test case tensor metadata
            For example, if a test case is { "mul_1d_1d": [base.Tensor([5]), base.Tensor([5])] }
            test_case_tensor is [base.Tensor([5]), base.Tensor([5])]
        '''

        input_list = []

        # ------ modify below for your model FROM here -------#
        x_tensor = self.createTFInput(test_case_tensor[0], input_list)

        output_node = tf.arg_min(x_tensor, 0, name=tc_name)
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
