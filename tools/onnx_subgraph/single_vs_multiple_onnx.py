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

import onnxruntime as ort
import numpy as np
import os
import re
import argparse


class ModelInference:
    """
    This class is used to infer multiple onnx models.
    Parameters:
        model_path: Path to the model files.
        subgraphsiostxt_path: Path to the txt file that describes the structure of the model graph.
    Output:
        outputs[0]: Inference result from the model.
    Description:
        Subgraphsiostxt_path is a txt file that describes the structure of the model graph and is used to get input/output node names.
        The model_path contains paths to multiple onnx files.
        The load_sessions function will sort the onnx models in the model_path according to the order specified in subgraphsiostxt_path.
    """
    def __init__(self, model_path, subgraphsiostxt_path):
        self.model_path = model_path
        self.subgraphsiostxt_path = subgraphsiostxt_path


    def infer_single_onnx_model(model_file, input_data):
        session = ort.InferenceSession(model_file)
        outputs = session.run(None, input_data)
        output_names = [output.name for output in session.get_outputs()]
        output_dict = {name: output for name, output in zip(output_names, outputs)}
        return output_dict


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-s',
                            '--single',
                            default='./resnet-test.onnx',
                            help="set single ONNX model path")
    arg_parser.add_argument('-m',
                            '--multi',
                            default='./subgraphs/',
                            help="set split subgraph models path")
    arg_parser.add_argument('-n',
                            '--node',
                            default='./scripts/subgraphs_ios.txt',
                            help="set subgraphs node i/o information")
    args = arg_parser.parse_args()

    # Initialize ModelInference instance for inference
    model_inference = ModelInference(args.multi, args.node)

    # Default input data dictionary
    default_input_data = {
        "x": np.random.rand(1, 3, 256, 256).astype(np.float32),
    }

    # Perform inference using a single ONNX model
    output_single = ModelInference.infer_single_onnx_model(args.single, default_input_data)
    print("Single model inference completed!")
