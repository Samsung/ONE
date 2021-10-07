#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import subprocess
import argparse
import traceback

#
# This script compares the execution result of luci-interpreter with that of TFLite interpreter
#
# Basic usage:
#   eval_verifier.py --driver build/compiler/luci-eval-driver/luci_eval_driver
#           --model inception_v3
parser = argparse.ArgumentParser()
parser.add_argument('--driver', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
args = parser.parse_args()

driver = args.driver
tflite_model = args.model + ".tflite"
circle_model = args.model + ".circle"

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

# Execute luci interpreter.
subprocess.run(
    [
        driver, circle_model,
        str(num_inputs), circle_model + ".input", circle_model + ".output"
    ],
    check=True)

# Compare the results.
for idx in range(len(interpreter.get_output_details())):
    output_details = interpreter.get_output_details()[idx]
    output_data = np.fromfile(circle_model + ".output" + str(idx),
                              output_details["dtype"])
    shape_file = open(circle_model + ".output" + str(idx) + ".shape", 'r')
    output_shape = [int(i) for i in shape_file.read().split(',')]
    luci_output_data = np.reshape(output_data, output_shape)
    intp_output_data = interpreter.get_tensor(output_details["index"])
    try:
        if output_details["dtype"] == np.uint8:
            if np.allclose(luci_output_data, intp_output_data, rtol=0, atol=0) == False:
                raise SystemExit("Execution result of " + tflite_model +
                                 " does not match with " + circle_model)
        elif output_details["dtype"] == np.float32:
            if np.allclose(
                    luci_output_data, intp_output_data, rtol=1.e-5, atol=1.e-5) == False:
                raise SystemExit("Execution result of " + tflite_model +
                                 " does not match with " + circle_model)
        elif output_details["dtype"] == np.int64:
            if np.allclose(luci_output_data, intp_output_data, rtol=0, atol=0) == False:
                raise SystemExit("Execution result of " + tflite_model +
                                 " does not match with " + circle_model)
        elif output_details["dtype"] == np.int32:
            if np.allclose(luci_output_data, intp_output_data, rtol=0, atol=0) == False:
                raise SystemExit("Execution result of " + tflite_model +
                                 " does not match with " + circle_model)
        else:
            raise SystemExit("Unsupported data type: ", output_details["dtype"])
    except:
        print(traceback.format_exc())
        quit(255)

quit(0)
