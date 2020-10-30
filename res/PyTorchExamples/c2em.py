#!/usr/bin/env python

# Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

# Caffe2 Example Manager

import caffe2.python.onnx.frontend as c2f
import onnx
import onnx_tf
import tensorflow as tf
import importlib
import argparse

from caffe2.proto import caffe2_pb2

print("ONNX version=", onnx.__version__)
print("TF version=", tf.__version__)

parser = argparse.ArgumentParser(description='Process caffe2 python examples')

parser.add_argument('examples', metavar='EXAMPLES', nargs='+')

args = parser.parse_args()

output_folder = "./output/"

for example in args.examples:
    module = importlib.import_module("c2examples." + example)

    onnx_model = c2f.caffe2_net_to_onnx_model(
        module._model_._net, module._model_init_._net, module._value_info_)
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, output_folder + example + ".onnx")

    onnx_model = onnx.load(output_folder + example + ".onnx")
    onnx.checker.check_model(onnx_model)

    tf_prep = onnx_tf.backend.prepare(onnx_model)
    tf_prep.export_graph(path=output_folder + example + ".TF")
    print("Generate '" + example + " TF' - Done")

    # for testing...
    converter = tf.lite.TFLiteConverter.from_saved_model(output_folder + example + ".TF")
    converter.allow_custom_ops = True
    converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    tflite_model = converter.convert()
    open(output_folder + example + ".tflite", "wb").write(tflite_model)
    print("Generate '" + example + ".tflite' - Done")
