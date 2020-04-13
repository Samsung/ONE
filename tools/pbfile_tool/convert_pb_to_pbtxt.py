# Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#
# This tool converts frozen pb file in tensorflow to pbtxt text file
#

import os
import argparse

from tensorflow.python.platform import gfile
import tensorflow as tf


def convert(pb_path):

    directory = os.path.dirname(pb_path)
    filename = os.path.basename(pb_path)

    with gfile.GFile(pb_path, 'rb') as f:
        content = f.read()

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(content)
    tf.import_graph_def(graph_def, name='')

    tf.train.write_graph(graph_def, directory, filename + '.pbtxt', as_text=True)

    return os.path.join(directory, filename + '.pbtxt')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='convert pb (binary) file to pbtxt (text) file')
    parser.add_argument("path", help="path of Tensorflow frozen model file in .pb format")

    args = parser.parse_args()
    pb_path = args.path

    pbtxt_path = convert(pb_path)

    print("converted: " + pbtxt_path)
