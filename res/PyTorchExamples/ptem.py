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

# PyTorch Example manager

import torch
import onnx
import onnx_tf
import tensorflow as tf
import importlib
import argparse

from pathlib import Path

print("PyTorch version=", torch.__version__)
print("ONNX version=", onnx.__version__)
print("ONNX-TF version=", onnx_tf.__version__)
print("TF version=", tf.__version__)

parser = argparse.ArgumentParser(description='Process PyTorch python examples')

parser.add_argument('examples', metavar='EXAMPLES', nargs='+')

args = parser.parse_args()

output_folder = "./output/"

Path(output_folder).mkdir(parents=True, exist_ok=True)

for example in args.examples:
    # load example code
    module = importlib.import_module("examples." + example)

    # save .pth
    torch.save(module._model_, output_folder + example + ".pth")
    print("Generate '" + example + ".pth' - Done")

    torch.onnx.export(
        module._model_, module._dummy_, output_folder + example + ".onnx", verbose=True)
    print("Generate '" + example + ".onnx' - Done")

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
