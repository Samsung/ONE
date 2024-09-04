#!/usr/bin/env python3
import numpy as np
import subprocess
import argparse
import traceback
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
    raise SystemExit("Unsupported dtype from file", dtype_str)


parser = argparse.ArgumentParser()
parser.add_argument('--driver', type=str, required=True)
parser.add_argument('--model_ref', type=str, required=True)
parser.add_argument('--work_path', type=str, required=True)
parser.add_argument('--rtolf32', type=str, required=False)
parser.add_argument('--atolf32', type=str, required=False)
args = parser.parse_args()

driver = args.driver
circle_model_ref = args.model_ref + ".circle"
circle_model = args.work_path + ".circle"
# circle_model is used as to follow existing luci_eval_verifier.py

rtolf32 = 1e-5
atolf32 = 1e-5
# NOTE reuse f32 value as int value too
rtolint = 0
atolint = 0
try:
    if args.rtolf32 != None:
        rtolf32 = float(args.rtolf32)
        rtolint = int(rtolf32)
    if args.atolf32 != None:
        atolf32 = float(args.atolf32)
        atolint = int(atolf32)
except ValueError:
    print("rtolf32 or atolf32 is not a number")
    quit(128)

# get num of inputs by checking existance of model.inputN
check_input = 0
while True:
    input_file_path = circle_model_ref + ".input" + str(check_input)
    if not os.path.isfile(input_file_path):
        num_inputs = check_input
        break
    check_input = check_input + 1

if num_inputs == 0:
    print("input file not exist for", circle_model_ref)
    quit(128)

# get num of outputs by checking existance of model.outputN
check_output = 0
while True:
    output_file_path = circle_model_ref + ".output" + str(check_output)
    if not os.path.isfile(output_file_path):
        num_outputs = check_output
        break
    check_output = check_output + 1

if num_outputs == 0:
    print("output file not exist for", circle_model_ref)
    quit(128)

# Execute luci interpreter with reference input
subprocess.run([
    driver, circle_model_ref,
    str(num_inputs), circle_model_ref + ".input", circle_model + ".output"
],
               check=True)

# Compare the results.
for idx in range(num_outputs):
    output_dtype = dtype_from_file(circle_model_ref + ".output" + str(idx) + ".dtype")
    shape_file = open(circle_model_ref + ".output" + str(idx) + ".shape", 'r')
    output_shape = [int(i) for i in shape_file.read().split(',')]

    output_data_ref = np.fromfile(circle_model_ref + ".output" + str(idx), output_dtype)
    luci_output_data_ref = np.reshape(output_data_ref, output_shape)

    output_data = np.fromfile(circle_model + ".output" + str(idx), output_dtype)
    luci_output_data = np.reshape(output_data, output_shape)

    try:
        if output_dtype == np.uint8:
            if np.allclose(luci_output_data,
                           luci_output_data_ref,
                           rtol=rtolint,
                           atol=atolint) == False:
                print("luci_output_data_ref", luci_output_data_ref)
                print("luci_output_data", luci_output_data)
                raise SystemExit("Execution result of " + circle_model_ref +
                                 " does not match with " + circle_model)
        elif output_dtype == np.float32:
            if np.allclose(luci_output_data,
                           luci_output_data_ref,
                           rtol=rtolf32,
                           atol=atolf32) == False:
                print("luci_output_data_ref", luci_output_data_ref)
                print("luci_output_data", luci_output_data)
                raise SystemExit("Execution result of " + circle_model_ref +
                                 " does not match with " + circle_model)
        elif output_dtype == np.int64:
            if np.allclose(luci_output_data,
                           luci_output_data_ref,
                           rtol=rtolint,
                           atol=atolint) == False:
                print("luci_output_data_ref", luci_output_data_ref)
                print("luci_output_data", luci_output_data)
                raise SystemExit("Execution result of " + circle_model_ref +
                                 " does not match with " + circle_model)
        elif output_dtype == np.int32:
            if np.allclose(luci_output_data,
                           luci_output_data_ref,
                           rtol=rtolint,
                           atol=atolint) == False:
                print("luci_output_data_ref", luci_output_data_ref)
                print("luci_output_data", luci_output_data)
                raise SystemExit("Execution result of " + circle_model_ref +
                                 " does not match with " + circle_model)
        elif output_dtype == np.int16:
            if np.allclose(luci_output_data,
                           luci_output_data_ref,
                           rtol=rtolint,
                           atol=atolint) == False:
                print("luci_output_data_ref", luci_output_data_ref)
                print("luci_output_data", luci_output_data)
                raise SystemExit("Execution result of " + circle_model_ref +
                                 " does not match with " + circle_model)
        elif output_dtype == np.bool_:
            if np.allclose(luci_output_data, luci_output_data_ref, rtol=0,
                           atol=0) == False:
                raise SystemExit("Execution result of " + circle_model_ref +
                                 " does not match with " + circle_model)
        else:
            raise SystemExit("Unsupported data type: ", output_dtype)
    except:
        print(traceback.format_exc())
        quit(255)

quit(0)
