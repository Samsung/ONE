#!/usr/bin/env python3
import h5py as h5
import numpy as np
import argparse
from io import StringIO
import os.path
import json
import logging
import sys

#
# This script checks if the min/max values recorded in the circle model are the same with the expected values
#
parser = argparse.ArgumentParser()
parser.add_argument(
    '--test_param',
    type=str,
    required=True,
    nargs='+',
    help=
    'All the param list to test. e.g. ${RECIPE_NAME_0} ${GRANULARITY_0} ${DTYPE_0} ${RECIPE_NAME_1} ${GRANULARITY_1} ${DTYPE_1}..'
)
parser.add_argument(
    '--bin_dir',
    type=str,
    required=True,
    help='Directory path wehre test files are generated')
parser.add_argument(
    '--source_dir',
    type=str,
    required=True,
    help='Directory path where expected outputs exist')
parser.add_argument('--mode', type=str, required=True, help='Mode to test')
args = parser.parse_args()

modes_to_expected_folder = {
    'fake_quantization': 'fake_quantization',
    'mixed_fake_quantization': 'fake_quantization',
    'record_minmax': 'record_minmax',
    'parallel_record_minmax': 'record_minmax',
    'quantization': 'quantization',
    'mixed_quantization': 'quantization',
    'weights_only_quantization': 'wo_quantization'
}
modes_to_input_h5_suffix = {
    'fake_quantization': 'fake_quantized.circle.h5',
    'mixed_fake_quantization': 'fake_quantized.mixed.circle.h5',
    'record_minmax': 'minmax_recorded.circle.h5',
    'parallel_record_minmax': 'parallel_minmax_recorded.circle.h5',
    'quantization': 'quantized.circle.h5',
    'mixed_quantization': 'quantized.mixed.circle.h5',
    'weights_only_quantization': 'wo_quantized.circle.h5'
}

test_param = args.test_param
bin_dir = args.bin_dir
source_dir = args.source_dir
mode = args.mode

log_format = '%(levelname)s: %(message)s'
formatter = logging.Formatter(log_format)
streamer = StringIO()
stream_handler = logging.StreamHandler(stream=streamer)
stream_handler.setFormatter(formatter)
logging.basicConfig(handlers=[stream_handler])

if mode not in modes_to_expected_folder.keys():
    raise SystemExit("Unsupported mode. --mode should be one of " +
                     str(modes_to_expected_folder.keys()))


def compare_fake_quantization(tensor, tensor_name, expect_dir):
    with open(expect_dir + "/" + tensor_name + ".json", "r") as expect_file:
        json_load = json.load(expect_file)
    expected_weights = np.array(json_load["weights"])
    input_weights = tensor["weights"][:]
    if np.allclose(input_weights, expected_weights, rtol=1.e-5, atol=1.e-5) == False:
        logging.error("Fake-quantized weights of " + tensor_name + " (" +
                      str(input_weights) + ") do not match with expected value (" +
                      str(expected_weights) + ").")
        return False
    return True


def compare_record_minmax(tensor, tensor_name, expect_dir):
    test_result = True
    with open(expect_dir + "/" + tensor_name + ".json", "r") as expect_file:
        json_load = json.load(expect_file)
    expected_min = np.array(json_load["min"])
    expected_max = np.array(json_load["max"])
    input_min = tensor["min"][:]
    input_max = tensor["max"][:]
    if np.allclose(input_min, expected_min, rtol=1.e-5, atol=1.e-5) == False:
        logging.error("Recorded min of " + tensor_name + " (" + str(input_min) +
                      ") does not match with expected value (" + str(expected_min) + ").")
        test_result = False
    if np.allclose(input_max, expected_max, rtol=1.e-5, atol=1.e-5) == False:
        logging.error("Recorded max of " + tensor_name + " (" + str(input_max) +
                      ") does not match with expected value (" + str(expected_max) + ").")
        test_result = False
    return test_result


def compare_quantization(tensor, tensor_name, expect_dir):
    test_result = True
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
                logging.error(
                    "Quantized weights of " + tensor_name + " (" + str(input_weights) +
                    ") do not match with expected value (" + str(expected_weights) + ").")
                test_result = False

        if key == "scale":
            expected_scale = np.array(json_load["scale"])
            input_scale = tensor["scale"][:]
            if np.allclose(input_scale, expected_scale, rtol=1.e-5, atol=1.e-5) == False:
                logging.error("Quantized scale of " + tensor_name + " (" +
                              str(input_scale) + ") do not match with expected value (" +
                              str(expected_scale) + ").")
                test_result = False

        if key == "zero_point":
            expected_zero_point = np.array(json_load["zero_point"])
            input_zero_point = tensor["zero_point"][:]
            if np.allclose(
                    input_zero_point, expected_zero_point, rtol=0, atol=1) == False:
                logging.error("Quantized zero_point of " + tensor_name + " (" +
                              str(input_zero_point) +
                              ") do not match with expected value (" +
                              str(expected_zero_point) + ").")
                test_result = False
    return test_result


def chunks(lst, n):
    """Yield successive n-sized chunks from the list"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


failed_log = dict()
inputs = test_param[0].split()
PARAM_SET_SIZE = 3
assert (len(inputs) % PARAM_SET_SIZE) == 0
inputs = list(chunks(inputs, PARAM_SET_SIZE))
for idx in range(len(inputs)):
    model_name = inputs[idx][0]
    granularity = inputs[idx][1]
    dtype = inputs[idx][2]

    testcase = f'{model_name}.{granularity}.{dtype}'
    test_result_file = os.path.join(bin_dir, testcase)
    input_h5 = f'{test_result_file}.{modes_to_input_h5_suffix[mode]}'
    with h5.File(input_h5, 'r') as input:
        for tensor_name in input.keys():
            expect_dir = f'{source_dir}/expected_outputs/{model_name}/{granularity}/{dtype}/{modes_to_expected_folder[mode]}'
            if os.path.isfile(expect_dir + "/" + tensor_name + ".json"):
                test_result = False
                if mode == "fake_quantization":
                    test_result = compare_fake_quantization(input[tensor_name],
                                                            tensor_name, expect_dir)
                elif mode == "record_minmax":
                    test_result = compare_record_minmax(input[tensor_name], tensor_name,
                                                        expect_dir)
                elif mode == "quantization":
                    test_result = compare_quantization(input[tensor_name], tensor_name,
                                                       expect_dir)
                elif mode == "weights_only_quantization":
                    # Assume weights have name "ker"
                    if tensor_name == "ker":
                        test_result = compare_quantization(input[tensor_name],
                                                           tensor_name, expect_dir)
                else:
                    raise SystemExit("Unsupproted mode.")

                if not test_result:
                    failed_log[testcase] = streamer.getvalue().rstrip()
                    # Clean log
                    streamer = StringIO()
                    stream_handler.setStream(streamer)

failed_number = len(failed_log)
if failed_number != 0:
    print('FAILED')
    for testcase in failed_log:
        print(f'- {testcase}')
        print(failed_log[testcase])

sys.exit(failed_number)
