import numpy as np
import tensorflow as tf
import subprocess
import os

#
# This script compares the execution result of luci-interpreter with that from ref_model path
#
# Basic usage:
#   luci_eval_verifier_ref.py --driver build/compiler/luci-eval-driver/luci_eval_driver
#           --ref_model ref_model_path --model this_model_path
# Assumption:
#   these file exist with its purpose
#   - ref_model_path.circle; circle model
#   - ref_model_path.circle.inputN; N'th input numpy data
#   - ref_model_path.circle.inputN.dtype; N'th input data type in text
#   - ref_model_path.circle.inputN.shape; N'th input data shape in CSV
#   - ref_model_path.circle.outputN; N'th output numpy data
#   - ref_model_path.circle.outputN.dtype; N'th output data type in text
#   - ref_model_path.circle.outputN.shape; N'th output data shape in CSV


def dtype_from_file(file_path):
    with open(file_path, 'r') as dtype_file:
        dtype_str = dtype_file.read()
    if dtype_str == "float32":
        return np.float32
    if dtype_str == "uint8":
        return np.uint8
    if dtype_str == "int16":
        return np.int16
    if dtype_str == "int32":
        return np.int32
    if dtype_str == "int64":
        return np.int64
    if dtype_str == "bool":
        return np.bool_
    assert False, "Unsupported dtype from file: " + dtype_str


def luci_eval_verify_ref(test_name,
                         ref_artifacts,
                         target_artifacts,
                         eval_driver,
                         rtolf32=1e-5,
                         atolf32=1e-5):
    circle_model_ref = os.path.join(ref_artifacts, test_name + ".circle")
    circle_model = os.path.join(target_artifacts, test_name + ".circle")

    # NOTE reuse f32 value as int value too
    rtolint = int(rtolf32)
    atolint = int(atolf32)

    # get num of inputs by checking existance of model.inputN
    check_input = 0
    while True:
        input_file_path = circle_model_ref + ".input" + str(check_input)
        if not os.path.isfile(input_file_path):
            num_inputs = check_input
            break
        check_input = check_input + 1

    assert num_inputs != 0, "input file not exist for " + circle_model_ref

    # get num of outputs by checking existance of model.outputN
    check_output = 0
    while True:
        output_file_path = circle_model_ref + ".output" + str(check_output)
        if not os.path.isfile(output_file_path):
            num_outputs = check_output
            break
        check_output = check_output + 1

    assert num_outputs != 0, "output file not exist for " + circle_model_ref

    # Execute luci interpreter with reference input
    subprocess.run(
        [
            eval_driver, circle_model_ref,
            str(num_inputs), circle_model_ref + ".input", circle_model + ".output"
        ],
        check=True)

    # Compare the results.
    for idx in range(num_outputs):
        output_dtype = dtype_from_file(circle_model_ref + ".output" + str(idx) + ".dtype")
        shape_file = open(circle_model_ref + ".output" + str(idx) + ".shape", 'r')
        output_shape = [int(i) for i in shape_file.read().split(',')]

        output_data_ref = np.fromfile(circle_model_ref + ".output" + str(idx),
                                      output_dtype)
        luci_output_data_ref = np.reshape(output_data_ref, output_shape)

        output_data = np.fromfile(circle_model + ".output" + str(idx), output_dtype)
        luci_output_data = np.reshape(output_data, output_shape)

        if output_dtype == np.uint8:
            assert np.allclose(
                luci_output_data, luci_output_data_ref, rtol=rtolint, atol=atolint
            ), "Execution result of " + circle_model_ref + " does not match with " + circle_model
        elif output_dtype == np.float32:
            assert np.allclose(
                luci_output_data, luci_output_data_ref, rtol=rtolf32, atol=atolf32
            ), "Execution result of " + circle_model_ref + " does not match with " + circle_model
        elif output_dtype == np.int64:
            assert np.allclose(
                luci_output_data, luci_output_data_ref, rtol=rtolint, atol=atolint
            ), "Execution result of " + circle_model_ref + " does not match with " + circle_model
        elif output_dtype == np.int32:
            assert np.allclose(
                luci_output_data, luci_output_data_ref, rtol=rtolint, atol=atolint
            ), "Execution result of " + circle_model_ref + " does not match with " + circle_model
        elif output_dtype == np.int16:
            assert np.allclose(
                luci_output_data, luci_output_data_ref, rtol=rtolint, atol=atolint
            ), "Execution result of " + circle_model_ref + " does not match with " + circle_model
        elif output_dtype == np.bool_:
            assert np.allclose(
                luci_output_data, luci_output_data_ref, rtol=0, atol=0
            ), "Execution result of " + circle_model_ref + " does not match with " + circle_model
        else:
            assert False, "Unsupported data type: " + output_dtype


# arguments must be in sync with `conftest.py`
def test_luci_eval_ref(default_ref_test_name: str, ref_artifacts_path: str,
                       target_artifacts_path: str, eval_driver_path: str):
    luci_eval_verify_ref(default_ref_test_name, ref_artifacts_path, target_artifacts_path,
                         eval_driver_path)


# arguments must be in sync with `conftest.py`
def test_luci_eval_tol_ref(tol_ref_test_name: str, ref_artifacts_path: str,
                           target_artifacts_path: str, eval_driver_path: str,
                           rtolf32: str, atolf32: str):
    luci_eval_verify_ref(tol_ref_test_name, ref_artifacts_path, target_artifacts_path,
                         eval_driver_path, float(rtolf32), float(atolf32))
