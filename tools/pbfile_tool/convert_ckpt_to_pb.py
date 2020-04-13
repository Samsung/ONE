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

# this file is added by NNFW to convert checkpoint file
# to frozen pb file
# and generate tensorboard for convenience

import os
import argparse
import tensorflow as tf
import model_freezer_util as util


def convert(checkpoint_dir, checkpoint_file_path):

    meta_path = os.path.join(checkpoint_file_path + '.meta')  # Your .meta file
    output_node_name = 'Model/concat'
    output_node_names = [output_node_name]  # Output nodes

    with tf.Session() as sess:

        # Restore the graph
        saver = tf.train.import_meta_graph(meta_path)

        # Load weights
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

        # save the graph into pb
        saved_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, output_node_names)

        pb_path = os.path.join(checkpoint_dir, 'graph.pb')
        with open(pb_path, 'wb') as f:
            f.write(saved_graph_def.SerializeToString())

    # freeze
    (frozen_pb_path, frozen_pbtxt_path) = util.freezeGraph(pb_path, checkpoint_file_path,
                                                           output_node_name)

    print("Freeze() Finished. Created :")
    print("\t-{}\n\t-{}\n".format(frozen_pb_path, frozen_pbtxt_path))

    # tensor board
    tensorboardLogDir = util.generateTensorboardLog([frozen_pb_path], [''],
                                                    os.path.join(
                                                        checkpoint_dir, ".tensorboard"))

    print("")
    print(
        "\t# Tensorboard: You can view original graph and frozen graph with tensorboard.")
    print("\t  Run the following:")
    print("\t  $ tensorboard --logdir={} ".format(tensorboardLogDir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='convert checkpoint file to pb file and freeze the pb file')
    parser.add_argument(
        "checkpoint_dir",
        help=
        "directory where checkpoint files are located. pb, pbtxt will also be generated into this folder."
    )
    parser.add_argument("checkpoint_file_name", help="name of checkpoint file")

    args = parser.parse_args()
    checkpoint_dir = args.checkpoint_dir
    checkpoint_file_path = os.path.join(checkpoint_dir, args.checkpoint_file_name)

    convert(checkpoint_dir, checkpoint_file_path)
