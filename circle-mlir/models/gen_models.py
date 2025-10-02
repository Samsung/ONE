#!/usr/bin/env python

# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

# This script will open folder in 'unit' or 'net' and produce ONNX model.
# refer circle-mlir/tools-test/gen-onnx/run_gen_onnx.py
# Usage example)
#   python3 gen_models.py Add_F32_R4 Add_F32_R4_C1

import torch
import onnx
import importlib
import argparse
import glob

from pathlib import Path

print("PyTorch version=", torch.__version__)
print("ONNX version=", onnx.__version__)


def load_module(model):
    # load model code in 'unit' folder
    module = None
    model_init_path = Path("./unit") / model / "__init__.py"
    model_name = "unit." + model
    if not model_init_path.exists():
        model_init_path = Path("./net") / model / "__init__.py"
        model_name = "net." + model
        if not model_init_path.exists():
            print("model of " + model + " not found.")
            return None
    try:
        module = importlib.import_module(model_name)
    except Exception as e:
        print("Error:", e)
        return None

    return module


def generate_pth(output_folder, model, module):
    # save .pth
    torch.save(module._model_, output_folder + model + ".pth")
    print("Generate '" + model + ".pth' - Done")


def generate_onnx(output_folder, model, module):
    opset_version = 14
    if hasattr(module._model_, 'onnx_opset_version'):
        opset_version = module._model_.onnx_opset_version()

    onnx_model_path = output_folder + model + ".onnx"

    m_keys = module.__dict__.keys()
    if '_io_names_' in m_keys and '_dynamic_axes_' in m_keys:
        torch.onnx.export(module._model_,
                          module._inputs_,
                          onnx_model_path,
                          input_names=module._io_names_[0],
                          output_names=module._io_names_[1],
                          dynamic_axes=module._dynamic_axes_,
                          opset_version=opset_version)
    else:
        torch.onnx.export(module._model_,
                          module._inputs_,
                          onnx_model_path,
                          opset_version=opset_version)

    if hasattr(module._model_, 'post_process'):
        module._model_.post_process(onnx_model_path)

    # check and run shape inference
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    inferred_model = onnx.shape_inference.infer_shapes(onnx_model)
    onnx.checker.check_model(inferred_model)
    onnx.save(inferred_model, onnx_model_path)

    print("Generate '" + model + ".onnx' - Done")


def generate_models(models):
    output_folder = "./output/"
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    for model in models:
        module = load_module(model)
        if module != None:
            generate_pth(output_folder, model, module)
            generate_onnx(output_folder, model, module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process PyTorch model')
    parser.add_argument('models', metavar='MODELS', nargs='+')
    args = parser.parse_args()
    models = args.models

    if len(models) == 1 and models[0] == '@':
        # generate for all models in unit and net folder
        globs = [f for f in glob.glob('./unit/*')]
        globs.sort()
        models_unit = [Path(f).stem for f in globs]

        globs = [f for f in glob.glob('./net/*')]
        globs.sort()
        models_net = [Path(f).stem for f in globs]

        models = models_unit + models_net

        generate_models(models)
    else:
        generate_models(models)
