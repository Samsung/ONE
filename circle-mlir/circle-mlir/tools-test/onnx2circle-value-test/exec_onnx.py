#!/usr/bin/env python3

import sys
import json
import numpy as np
import h5py
import onnx
import onnxruntime as rt

from pathlib import Path

# local python files
import util_h5_file


def get_np_type(onnx_type):
    if onnx_type == 'tensor(float)':
        return np.float32
    if onnx_type == 'tensor(int64)':
        return np.int64
    if onnx_type == 'tensor(bool)':
        return np.bool_
    # TODO support other types
    raise SystemExit('Unsupported onnx_type at get_np_type():' + onnx_type)


def fit_layout(tensor, is_nchw):
    if not is_nchw and tensor.ndim == 4:
        return np.transpose(tensor, (0, 2, 3, 1))
    return tensor


def h5_save_input(filepath, inputs, input_data, is_nchw=False):
    filename = util_h5_file.get_input_filename(filepath, is_nchw)
    h5f = h5py.File(filename, 'w')
    for i in range(len(inputs)):
        input_name = inputs[i].name
        input_data_values = fit_layout(input_data[input_name], is_nchw)
        input_name_res = str(i).zfill(5) + '-' + input_name
        h5f.create_dataset(input_name_res, data=input_data_values)

    h5f.close()


def h5_save_output(filepath, output_names, output_data, is_nchw=False):
    filename = util_h5_file.get_output_filename(filepath, is_nchw)
    h5f = h5py.File(filename, 'w')
    for i in range(len(output_names)):
        output_name = output_names[i]
        output_data_values = fit_layout(output_data[i], is_nchw)
        output_name_res = str(i).zfill(5) + '-' + output_name
        h5f.create_dataset(output_name_res, data=output_data_values)

    h5f.close()


def get_input_limit(input, default_low=0, default_high=100):
    low = default_low
    high = default_high
    # NOTE each node may have `doc_string` field, where simple documentation can be placed
    # referenced: https://github.com/onnx/onnx/blob/main/docs/IR.md
    if input.HasField("doc_string") and input.doc_string:
        try:
            # NOTE this json is a private format, created to specify a range of input values
            # low : lower limit of the input value (inclusive)
            # high : upper limit of the input value (exclusive)
            decoded_json = json.loads(input.doc_string)
            low = decoded_json.get('low', default_low)
            high = decoded_json.get('high', default_high)
        except json.JSONDecodeError:
            return default_low, default_high
    return low, high


def exec_model(filepath, is_nchw=False):
    options = rt.SessionOptions()
    # NOTE this is needed for U18.04
    # referenced: https://github.com/microsoft/onnxruntime/issues/10113
    options.intra_op_num_threads = 4
    # NOTE set `providers` for https://github.com/microsoft/onnxruntime/issues/17631
    providers = rt.get_available_providers()
    session = rt.InferenceSession(filepath, sess_options=options, providers=providers)
    onnx_model = onnx.load(filepath)

    inputs = [input for input in session.get_inputs()]
    input_data = dict()
    for i in range(len(inputs)):
        input_name = inputs[i].name
        np_type = get_np_type(inputs[i].type)
        if np_type == np.float32:
            random_data = np.random.random(inputs[i].shape).astype(np_type)
        elif np_type == np.uint8:
            random_data = np.random.randint(0, 256, size=inputs[i].shape).astype(np_type)
        elif np_type == np.int64:
            low, high = get_input_limit(onnx_model.graph.input[i], 0, 100)
            random_data = np.random.randint(low, high,
                                            size=inputs[i].shape).astype(np_type)
        elif np_type == np.bool_:
            random_data = np.random.choice(a=[True, False],
                                           size=inputs[i].shape).astype(np_type)
        else:
            raise SystemExit('Unsupported input dtype')

        input_data[input_name] = random_data

    outputs = session.get_outputs()
    output_names = []
    for i in range(len(outputs)):
        output_names.append(outputs[i].name)

    # Run the model
    result = session.run(output_names, input_data)

    # Save I/O data to h5 file
    filename = filepath  # Path(filepath).name
    h5_save_input(filename, inputs, input_data, is_nchw)
    h5_save_output(filename, output_names, result, is_nchw)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        filepath = Path(sys.argv[0])
        sys.exit('Usage: ' + filepath.name + ' [model.onnx]')

    exec_model(sys.argv[1], True)
