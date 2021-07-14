#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import subprocess
import argparse
import traceback
import json

#
# This script compares the execution result of TFLite interpreter and
# partitioned model(s) from a circle model
#
# Basic usage for example:
#   part_eval_one.py \
#       --driver build/compiler/circle-part-driver/circle-part-driver \
#       --name test_file
#
parser = argparse.ArgumentParser()
parser.add_argument('--driver', type=str, required=True)
parser.add_argument('--name', type=str, required=True)
args = parser.parse_args()

driver = args.driver
tflite_model = args.name + ".tflite"
circle_model = args.name + ".circle"
partition_conn_ini = args.name + ".conn.ini"
partition_conn_json = args.name + ".conn.json"
expected_count = args.name + ".excnt"

# Check expected count of models from partitioning
try:
    with open(expected_count, "r") as expected_count_file:
        expected_count_line = expected_count_file.readline()

    expected_count_line = int(expected_count_line)
    if expected_count_line:
        with open(partition_conn_json) as json_file:
            json_data = json.load(json_file)
            parts_value = json_data["parts"]
            if len(parts_value) != expected_count_line:
                print("Partitioned model count differs from expected:",
                      expected_count_line)
                quit(255)

            print("Partitioned model count expected: ", expected_count_line)
    else:
        print("Skip expected partitioned model count check: 0")

except:
    print("Skip expected partitioned model count check: error")

# Build TFLite interpreter.
interpreter = tf.lite.Interpreter(tflite_model)
interpreter.allocate_tensors()

# Generate random input data.
num_inputs = len(interpreter.get_input_details())
for i in range(num_inputs):
    input_details = interpreter.get_input_details()[i]
    if input_details["dtype"] == np.float32:
        input_data = np.array(
            np.random.random_sample(input_details["shape"]), input_details["dtype"])
    elif input_details["dtype"] == np.uint8:
        input_data = np.array(
            np.random.randint(0, 256, size=input_details["shape"]),
            input_details["dtype"])
    elif input_details["dtype"] == np.bool_:
        input_data = np.array(
            np.random.choice(a=[True, False], size=input_details["shape"]),
            input_details["dtype"])
    else:
        raise SystemExit("Unsupported input dtype")

    interpreter.set_tensor(input_details["index"], input_data)
    input_data.tofile(circle_model + ".input" + str(i))

# Do inference
interpreter.invoke()

# Execute circle-part-driver.
partition_command = [
    driver, partition_conn_ini,
    str(num_inputs), circle_model + ".input", circle_model + ".output"
]
print("Run: ")
for arg in partition_command:
    print("    ", arg, "\\")
print("", flush=True)

subprocess.run(partition_command, check=True)

# Compare the results.
for idx in range(len(interpreter.get_output_details())):
    output_details = interpreter.get_output_details()[idx]
    output_data = np.fromfile(circle_model + ".output" + str(idx),
                              output_details["dtype"])
    shape_file = open(circle_model + ".output" + str(idx) + ".shape", 'r')
    output_shape = [int(i) for i in shape_file.read().split(',')]
    luci_output_data = np.reshape(output_data, output_shape)
    try:
        if output_details["dtype"] == np.uint8:
            if np.allclose(
                    luci_output_data,
                    interpreter.get_tensor(
                        interpreter.get_output_details()[idx]["index"]),
                    rtol=0,
                    atol=0) == False:
                raise SystemExit("Execution result of " + tflite_model +
                                 " does not match with " + circle_model)
        elif output_details["dtype"] == np.float32:
            if np.allclose(
                    luci_output_data,
                    interpreter.get_tensor(
                        interpreter.get_output_details()[idx]["index"]),
                    rtol=1.e-5,
                    atol=1.e-5) == False:
                raise SystemExit("Execution result of " + tflite_model +
                                 " does not match with " + circle_model)
        elif output_details["dtype"] == np.int64:
            if np.allclose(
                    luci_output_data,
                    interpreter.get_tensor(
                        interpreter.get_output_details()[idx]["index"]),
                    rtol=0,
                    atol=0) == False:
                raise SystemExit("Execution result of " + tflite_model +
                                 " does not match with " + circle_model)
        elif output_details["dtype"] == np.int32:
            if np.allclose(
                    luci_output_data,
                    interpreter.get_tensor(
                        interpreter.get_output_details()[idx]["index"]),
                    rtol=0,
                    atol=0) == False:
                raise SystemExit("Execution result of " + tflite_model +
                                 " does not match with " + circle_model)
        else:
            raise SystemExit("Unsupported data type: ", output_details["dtype"])
    except:
        print(traceback.format_exc())
        quit(255)

quit(0)
