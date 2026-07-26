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
from model_inference_multiple_output import *
import os


def compare_results(output_single, output_multiple):
    """
    Compares the Mean Squared Error (MSE) between identically named outputs from two inference result dictionaries.
    Ensures each output name is processed only once.
    """
    all_keys = set(output_single.keys()).union(set(output_multiple.keys()))
    for key in sorted(all_keys):
        if key in output_single and key in output_multiple:
            single_output = np.array(output_single[key])
            multiple_output = np.array(output_multiple[key])
            mse = np.mean((single_output - multiple_output)**2)
            print(f"Output '{key}' MSE: {mse}")
        else:
            print(f"Output '{key}' is missing in one of the result sets.")


def prepare_initial_input_data(onnx_model_path, default_input_data):
    """
    Prepares initial input data for inference.

    Args:
        onnx_model_path (str): Path to the ONNX model file.
        default_input_data (dict): Dictionary containing default input data.

    Returns:
        dict: Dictionary with user-specified or default shaped and typed input data.
    """
    session = ort.InferenceSession(onnx_model_path)
    input_info = {input.name: input.shape for input in session.get_inputs()}

    initial_input_data = {}
    dtype_map = {'f': np.float32, 'i': np.int64}

    for input_name, shape in input_info.items():
        custom_shape_str = input(
            f"Enter new shape for input '{input_name}' (comma-separated integers), or press Enter to use default: "
        )
        custom_dtype_str = input(
            f"Enter data type for input '{input_name}' ('f' for float32, 'i' for int64), or press Enter to use default: "
        )

        if not custom_shape_str:
            new_shape = default_input_data[input_name].shape
        else:
            try:
                new_shape = [int(dim) for dim in custom_shape_str.split(',')]
            except ValueError:
                print("Invalid input, please ensure you enter comma-separated integers.")
                continue

        if not custom_dtype_str:
            dtype = default_input_data[input_name].dtype
        else:
            dtype = dtype_map.get(custom_dtype_str.strip(), None)
            if dtype is None:
                print("Invalid data type, please enter 'f' or 'i'.")
                continue

        input_data = np.random.rand(*new_shape).astype(dtype)
        initial_input_data[input_name] = input_data

    return initial_input_data


# Define paths for single ONNX model and split subgraph models
single_onnx_model_path = './resnet-test.onnx'
model_path = './subgraphs/'
subgraphsiostxt_path = './subgraphs_ios.txt'

# Initialize ModelInference instance for inference
model_inference = ModelInference(model_path, subgraphsiostxt_path)

# Default input data dictionary
default_input_data = {
    "x": np.random.rand(1, 3, 256, 256).astype(np.float32),
}

#initial_input_data = prepare_initial_input_data(single_onnx_model_path, default_input_data)
initial_input_data = default_input_data

# Perform inference using a single ONNX model
output_single = ModelInference.infer_single_onnx_model(single_onnx_model_path,
                                                       initial_input_data)
print("Single model inference completed!")

# Retrieve all output names from the single model
output_names_list = list(output_single.keys())

# Perform inference using multiple split subgraph models
output_multiple = model_inference.inference(initial_input_data, output_names_list)
print("Multiple subgraph inference completed!")

print("Comparing inference results between single ONNX model and multiple subgraphs...")
compare_results(output_single, output_multiple)
