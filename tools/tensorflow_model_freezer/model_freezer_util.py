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

# utility for nncc

import os
import sys

import tensorflow as tf
from google.protobuf import text_format
from tensorflow.python.platform import gfile
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib


# --------
def file_validity_check(fn, ext_must_be=''):
    ''' check if file exist and file extention is corrent '''
    if os.path.exists(fn) == False:
        print("# error: file does not exist " + fn)
        return False

    if ext_must_be != '':
        ext = os.path.splitext(fn)[1]
        if ext[1:].lower(
        ) != ext_must_be:  # ext contains , e.g., '.pb'. need to exclud '.'
            print("# error: wrong extension {}. Should be {} ".format(ext, ext_must_be))
            return False

    return True


# --------
def importGraphIntoSession(sess, filename, graphNameAfterImporting):
    # this should be called inside
    # with tf.Session() as sess:
    assert sess
    (_, _, ext) = splitDirFilenameExt(filename)
    if (ext.lower() == 'pb'):
        with gfile.FastGFile(filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

    elif (ext.lower() == 'pbtxt'):
        with open(filename, 'r') as reader:
            graph_def = tf.GraphDef()
            text_format.Parse(reader.read(), graph_def)
    else:
        print("# Error: unknown extension - " + ext)

    tf.import_graph_def(graph_def, name=graphNameAfterImporting)


# --------
def splitDirFilenameExt(path):
    # in case of '/tmp/.ssh/my.key.dat'
    # this returns ('/tmp/.ssh', 'my.key', 'dat')
    directory = os.path.split(path)[0]
    ext = os.path.splitext(path)[1][1:]  # remove '.', e.g., '.dat' -> 'dat'
    filename = os.path.splitext(os.path.split(path)[1])[0]

    return (directory, filename, ext)


# --------
def convertPbtxt2Pb(pbtxtPath):
    ''' convert pbtxt file to pb file. e.g., /tmp/a.pbtxt --> /tmp/a.pb '''
    with open(pbtxtPath) as f:
        txt = f.read()

    gdef = text_format.Parse(txt, tf.GraphDef())

    (directory, filename, ext) = splitDirFilenameExt(pbtxtPath)

    tf.train.write_graph(gdef, directory, filename + '.pb', as_text=False)

    return os.path.join(directory, filename + '.pb')


# --------
def convertPb2Pbtxt(pbPath):
    ''' convert pb file to pbtxt file. e.g., /tmp/a.pb --> /tmp/a.pbtxt '''

    (directory, filename, ext) = splitDirFilenameExt(pbPath)

    with gfile.FastGFile(pbPath, 'rb') as f:
        content = f.read()

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(content)
    tf.import_graph_def(graph_def, name='')

    tf.train.write_graph(graph_def, directory, filename + '.pbtxt', as_text=True)

    return os.path.join(directory, filename + '.pbtxt')


# --------
def savePbAndCkpt(sess, directory, fn_prefix):
    ''' save files related to session's graph into directory.
        - fn_prefix.pb : binary protocol buffer file
        - fn_prefix.pbtxt : text format of protocol buffer file
        - fn_prefix.ckpt.* : checkpoing files contains values of variables

        returns (path of pb file, path of pbtxt file, path of ckpt files)
    '''

    tf.train.write_graph(sess.graph_def, directory, fn_prefix + '.pb', as_text=False)
    tf.train.write_graph(sess.graph_def, directory, fn_prefix + '.pbtxt', as_text=True)

    # save a checkpoint file, which will store the above assignment
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(directory, 'checkoiint', fn_prefix + '.ckpt'))

    return (os.path.join(directory,
                         fn_prefix + '.pb'), os.path.join(directory,
                                                          fn_prefix + '.pbtxt'),
            os.path.join(directory, 'checkoiint', fn_prefix + '.ckpt'))


def optimizeGraph(input_graph_path, input_node_name, output_node_name):
    ''' this function calls optimize_for_inference of tensorflow and generates '*_optimized.pb'.

      - input_graph_path : must be a path to pb file
      - input_node_name  : name of input operation node
      - output_node_name : name of head(top) operation node
    '''

    (directory, fn, ext) = splitDirFilenameExt(input_graph_path)
    output_optimized_graph_path = os.path.join(directory, fn + '_optimized.pb')

    # Optimize for inference
    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(input_graph_path, "rb") as f:
        data = f.read()
        input_graph_def.ParseFromString(data)
        output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_name.split(","), output_node_name.split(","),
            tf.float32.as_datatype_enum)

    # Save the optimized graph
    f = tf.gfile.FastGFile(output_optimized_graph_path, "w")
    f.write(output_graph_def.SerializeToString())

    return output_optimized_graph_path


# --------
def freezeGraph(input_graph_path, checkpoint_path, output_node_name):
    ''' this function calls freeze_grapy.py of tensorflow and generates '*_frozen.pb' and '*_frozen.pbtxt'.

        - input_graph_path : must be a path to pb file
        - checkpoint_path  : path of *.ckpt, e.g., '/tmp/inception_v3/graph.ckpt'
        - output_node_name : name of head(top) operation node
        '''

    input_saver_def_path = ""
    input_binary = True

    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    clear_devices = True

    (directory, fn, ext) = splitDirFilenameExt(input_graph_path)
    output_frozen_graph_path = os.path.join(directory, fn + '_frozen.pb')

    if file_validity_check(input_graph_path, 'pb') == False:
        print("Error: {} not found or not have pb extension".format(input_graph_path))
        sys.exit(0)

    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path, input_binary,
                              checkpoint_path, output_node_name, restore_op_name,
                              filename_tensor_name, output_frozen_graph_path,
                              clear_devices, "")

    pbtxtPath = convertPb2Pbtxt(output_frozen_graph_path)

    return (output_frozen_graph_path, pbtxtPath)


# --------
def generateTensorboardLog(pbFiles, graphNames, directory):
    ''' Generate logs for tensorboard. after calling this, graph(s) can be viewed inside tensorboard.
        This function creates a new Session(), so call this outside of 'with Session():'

        parameters:
        - pbFiles: if multiple graphs needs to be shown, pass the list of pb (or pbtxt) files
        - directory: parent directory of '/.tensorboard' directory where log files are saved

        how to run tensorboard:
              $ tensorboard --logdir=directory_in_parameter
    '''
    assert len(pbFiles) == len(graphNames)

    # without this, graph used previous session is reused : https://stackoverflow.com/questions/42706761/closing-session-in-tensorflow-doesnt-reset-graph
    tf.reset_default_graph()
    with tf.Session() as sess:

        i = 0
        for pbFile in pbFiles:
            graphName = graphNames[i]
            importGraphIntoSession(sess, pbFile, graphName)
            i = i + 1

    tbLogPath = directory
    train_writer = tf.summary.FileWriter(tbLogPath)
    train_writer.add_graph(sess.graph)
    train_writer.flush()
    train_writer.close()

    return tbLogPath


#--------
def isScalar(x):
    '''
    keyword argument:
    x - base_freezer.Tensor
    '''

    return (type(x.getShape()) == [])
