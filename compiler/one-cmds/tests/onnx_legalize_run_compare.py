# Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

import onnxruntime as rt
import onnx
import sys
import numpy as np
import importlib.util


def _generate_inputs(model):
    """Generate random inputs for given model

    Args:
        model (onnx.onnx_ml_pb2.ModelProto): target model

    Returns:
        dict from str to numpy.ndarray: generated inputs
    """
    inputs = {}
    for input in model.graph.input:
        # check if elem type is float32
        # list of types could be extended, this is a property of current testsuite
        assert (
            input.type.tensor_type.elem_type == onnx.TensorProto.DataType.Value("FLOAT"))
        input_shape = []
        for dim in input.type.tensor_type.shape.dim:
            input_shape += [dim.dim_value]
        inputs[input.name] = np.random.random(input_shape).astype(np.float32)
    return inputs


def _run_model(model, inputs):
    """Run given model

    Args:
        model (onnx.onnx_ml_pb2.ModelProto): target model
        inputs (dict from str to numpy.ndarray): sample inputs

    Returns:
        list of numpy.ndarray: inference outputs
    """
    output_names = list(map(lambda output: output.name, model.graph.output))
    options = rt.SessionOptions()
    # NOTE this is needed for U18.04
    # referenced: https://github.com/microsoft/onnxruntime/issues/10113
    options.intra_op_num_threads = 4
    # NOTE set `providers` for https://github.com/microsoft/onnxruntime/issues/17631
    providers = rt.get_available_providers()
    session = rt.InferenceSession(
        model.SerializeToString(), sess_options=options, providers=providers)
    outputs = session.run(output_names, inputs)
    return outputs


def _compare_results(ref_outputs, test_outputs, tolerance):
    """Generate random inputs for given model

    Args:
        ref_outputs (list of numpy.ndarray): reference values (original model results)
        test_outputs (list of numpy.ndarray): tested values (modified model results)
        tolerance (float): maximum acceptable relative difference

    Returns:
        bool: True if outputs considered equal, False otherwise
    """
    num_outputs = len(ref_outputs)
    assert (len(test_outputs) == num_outputs)
    for i in range(num_outputs):
        if ref_outputs[i].shape != test_outputs[i].shape:
            print("output {} shape mismatch: ref({}) vs test({})".format(
                i, ref_outputs[i].shape, test_outputs[i].shape))
            return False

        abs_difference = np.abs(ref_outputs[i] - test_outputs[i])
        abs_ref_maximum = np.abs(ref_outputs[i]).max()
        peak_error = abs_difference.max() / abs_ref_maximum

        if peak_error > tolerance:
            print("output {} peak error to value ratio {} is too big".format(
                i, peak_error))
            return False
    return True


if __name__ == '__main__':
    if len(sys.argv) < 6:
        exit('expecting 5 arguments:\n'
             '  - path to input model\n'
             '  - path to "legalized" model\n'
             '  - path to onnx_legalizer.py\n'
             '  - base name for generated test inputs\n'
             '  - output tolerance')
    input_model_path = sys.argv[1]
    output_model_path = sys.argv[2]
    onnx_legalizer_path = sys.argv[3]
    input_dump_path = sys.argv[4]
    tolerance = float(sys.argv[5])

    onnx_legalizer_spec = importlib.util.spec_from_file_location(
        "onnx_legalizer", onnx_legalizer_path)
    onnx_legalizer = importlib.util.module_from_spec(onnx_legalizer_spec)
    onnx_legalizer_spec.loader.exec_module(onnx_legalizer)

    model = onnx.load(input_model_path)

    inputs = _generate_inputs(model)

    for i in inputs:
        np.save('{}_{}.npy'.format(input_dump_path, i), inputs[i])

    ref_outputs = _run_model(model, inputs)

    options = onnx_legalizer.LegalizeOptions()
    options.unroll_rnn = True
    options.unroll_lstm = True
    onnx_legalizer.legalize(model, options)

    with open(output_model_path, 'wb') as f:
        f.write(model.SerializeToString())

    test_outputs = _run_model(model, inputs)

    if not _compare_results(ref_outputs, test_outputs, tolerance):
        exit('comparison failed')
