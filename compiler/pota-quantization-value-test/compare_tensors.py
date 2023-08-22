#!/usr/bin/env python3
import h5py as h5
import numpy as np
import argparse
import os.path
import json
import sys

#
# This script checks if the min/max values recorded in the circle model are the same with the expected values
#
# Basic usage:
#   compare_tensors.py --input_h5 <path/to/iput/h5> --expect_dir <path/to/expect/dir> --mode <compare_mode>
#   ex: compare_minmax.py --input_h5 Add_000.h5 --expect_dir expected_outputs/Add_000 --mode fake_quantization

parser = argparse.ArgumentParser()
parser.add_argument('--input_h5', type=str, required=True)
parser.add_argument('--expect_dir', type=str, required=True)
parser.add_argument('--mode', type=str, required=True)
args = parser.parse_args()

supported_modes = [
    "fake_quantization", "record_minmax", "quantization", "weights_only_quantization"
]

model = args.input_h5
expect_dir = args.expect_dir
mode = args.mode

failed_cases = 0

if mode not in supported_modes:
    raise SystemExit("Unsupported mode. --mode should be one of " + str(supported_modes))


def compare_fake_quantization(tensor, tensor_name, expect_dir):
    global failed_cases
    with open(expect_dir + "/" + tensor_name + ".json", "r") as expect_file:
        json_load = json.load(expect_file)
    expected_weights = np.array(json_load["weights"])
    input_weights = tensor["weights"][:]
    if np.allclose(input_weights, expected_weights, rtol=1.e-5, atol=1.e-5) == False:
        print("Fake-quantized weights of " + tensor_name + " (" + str(input_weights) +
              ") do not match with expected value (" + str(expected_weights) + ").")
        failed_cases += 1


def compare_record_minmax(tensor, tensor_name, expect_dir):
    global failed_cases
    with open(expect_dir + "/" + tensor_name + ".json", "r") as expect_file:
        json_load = json.load(expect_file)
    expected_min = np.array(json_load["min"])
    expected_max = np.array(json_load["max"])
    input_min = tensor["min"][:]
    input_max = tensor["max"][:]
    if np.allclose(input_min, expected_min, rtol=1.e-5, atol=1.e-5) == False:
        print("Recorded min of " + tensor_name + " (" + str(input_min) +
              ") does not match with expected value (" + str(expected_min) + ").")
        failed_cases += 1
    if np.allclose(input_max, expected_max, rtol=1.e-5, atol=1.e-5) == False:
        print("Recorded max of " + tensor_name + " (" + str(input_max) +
              ") does not match with expected value (" + str(expected_max) + ").")
        failed_cases += 1


def compare_quantization(tensor, tensor_name, expect_dir):
    global failed_cases
    with open(expect_dir + "/" + tensor_name + ".json", "r") as expect_file:
        json_load = json.load(expect_file)
    for key in json_load:
        if key == "weights":
            expected_weights = np.array(json_load["weights"])
            input_weights = tensor["weights"][()]
            abs_tolerance = 1
            # We use higher tolerance for int64 data (bias of int16-quantized model)
            if tensor["weights"].dtype == 'int64':
                abs_tolerance = 5

            if np.allclose(
                    input_weights, expected_weights, rtol=0, atol=abs_tolerance) == False:
                print("Quantized weights of " + tensor_name + " (" + str(input_weights) +
                      ") do not match with expected value (" + str(expected_weights) +
                      ").")
                failed_cases += 1

        if key == "scale":
            expected_scale = np.array(json_load["scale"])
            input_scale = tensor["scale"][:]
            if np.allclose(input_scale, expected_scale, rtol=1.e-5, atol=1.e-5) == False:
                print("Quantized scale of " + tensor_name + " (" + str(input_scale) +
                      ") do not match with expected value (" + str(expected_scale) + ").")
                failed_cases += 1

        if key == "zero_point":
            expected_zero_point = np.array(json_load["zero_point"])
            input_zero_point = tensor["zero_point"][:]
            if np.allclose(
                    input_zero_point, expected_zero_point, rtol=0, atol=1) == False:
                print("Quantized zero_point of " + tensor_name + " (" +
                      str(input_zero_point) + ") do not match with expected value (" +
                      str(expected_zero_point) + ").")
                failed_cases += 1


with h5.File(model, "r") as input:
    for tensor_name in input.keys():
        # We only check the given golden data
        if os.path.isfile(expect_dir + "/" + tensor_name + ".json"):
            print("Compare " + tensor_name)
            if mode == "fake_quantization":
                compare_fake_quantization(input[tensor_name], tensor_name, expect_dir)
            elif mode == "record_minmax":
                compare_record_minmax(input[tensor_name], tensor_name, expect_dir)
            elif mode == "quantization":
                compare_quantization(input[tensor_name], tensor_name, expect_dir)
            elif mode == "weights_only_quantization":
                # Assume weights have name "ker"
                if tensor_name == "ker":
                    compare_quantization(input[tensor_name], tensor_name, expect_dir)
            else:
                raise SystemExit("Unsupproted mode.")

sys.exit(failed_cases)
