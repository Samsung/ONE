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
        Subgraphsiostxt_path is a txt file that describes the structure of the model graph and
        is used to get input/output node names.The model_path contains paths to multiple onnx files.
        The load_sessions function will sort the onnx models in the model_path according to the 
        order specified in subgraphsiostxt_path.
    """
    def __init__(self, model_path, subgraphsiostxt_path):
        self.model_path = model_path
        self.subgraphsiostxt_path = subgraphsiostxt_path
        self.sessions, self.sorted_file_paths = self.load_sessions()

    def load_sessions(self):
        with open(self.subgraphsiostxt_path, 'r') as file:
            content = file.read()
        subgraph_order_map = {}
        matches = re.findall(r'(\w+)subgraph(\d+): order(\d+)', content)

        for match in matches:
            subgraph_type, subgraph_number, order = match
            # lower_subgraph_type = subgraph_type.lower()
            file_path = os.path.join(self.model_path,
                                     f"{subgraph_type}subgraph{subgraph_number}.onnx")
            if int(order) in subgraph_order_map:
                subgraph_order_map[int(order)].append(file_path)
            else:
                subgraph_order_map[int(order)] = [file_path]

        sorted_file_paths = []
        for order in sorted(subgraph_order_map.keys()):
            sorted_file_paths.extend(subgraph_order_map[order])

        sessions = [ort.InferenceSession(model) for model in sorted_file_paths]
        return sessions, sorted_file_paths

    def infer_multiple_onnx_models(self,
                                   initial_input_data,
                                   output_names_to_collect=None):
        if output_names_to_collect is None:
            return {}
        input_data = initial_input_data
        collected_outputs = {}

        for i, (session,
                model_file) in enumerate(zip(self.sessions, self.sorted_file_paths)):
            input_names = [inp.name for inp in session.get_inputs()]
            output_names = [out.name for out in session.get_outputs()]
            model_input_data = {name: input_data[name] for name in input_names}
            outputs = session.run(None, model_input_data)
            current_model_outputs = dict(zip(output_names, outputs))

            for output_name in output_names_to_collect:
                if output_name in current_model_outputs:
                    collected_outputs[output_name] = current_model_outputs[output_name]

            if i < len(self.sessions) - 1:
                input_data.update(current_model_outputs)
        return collected_outputs

    def infer_single_onnx_model(model_file, input_data):
        session = ort.InferenceSession(model_file)
        outputs = session.run(None, input_data)
        output_names = [output.name for output in session.get_outputs()]
        output_dict = {name: output for name, output in zip(output_names, outputs)}
        return output_dict


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
            f"Enter new shape for input '{input_name}' (comma-separated integers),\
            or press Enter to use default: ")
        custom_dtype_str = input(
            f"Enter data type for input '{input_name}' ('f' for float32, 'i' for int64),\
            or press Enter to use default: ")

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
                print("Invalid data type, set as default, please enter 'f' or 'i'.")
                dtype = default_input_data[input_name].dtype
                continue

        input_data = np.random.rand(*new_shape).astype(dtype)
        initial_input_data[input_name] = input_data

    return initial_input_data


def compare_results(output_single, output_multiple):
    """
    Compares the Mean Squared Error (MSE) between identically named outputs from
    two inference result dictionaries.Ensures each output name is processed only once.
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
    initial_input_data = prepare_initial_input_data(args.single, default_input_data)
    # Perform inference using a single ONNX model
    output_single = ModelInference.infer_single_onnx_model(args.single,
                                                           initial_input_data)
    print("Single model inference completed!")

    # Retrieve all output names from the single model
    output_names_list = list(output_single.keys())

    # Perform inference using multiple split subgraph models
    output_multiple = model_inference.infer_multiple_onnx_models(
        initial_input_data, output_names_list)
    print("Multiple subgraph inference completed!")

    print("Comparing inference results between single ONNX model \
          and multiple subgraphs...")
    compare_results(output_single, output_multiple)
