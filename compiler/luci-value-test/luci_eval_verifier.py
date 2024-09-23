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
parser.add_argument('--rtolf32', type=str, required=False)
parser.add_argument('--atolf32', type=str, required=False)
args = parser.parse_args()

driver = args.driver
tflite_model = args.model + ".tflite"
circle_model = args.model + ".circle"

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

# Build TFLite interpreter.
interpreter = tf.lite.Interpreter(tflite_model)
interpreter.allocate_tensors()

# Read SignatureDef and get output tensor id orders for remapping
full_signatures = interpreter._get_full_signature_list()
full_signatures_outputs_remap = None
if full_signatures != None:
    signature_serving_default = full_signatures.get('serving_default', None)
    if signature_serving_default != None:
        signature_outputs = signature_serving_default['outputs']

        full_signatures_outputs_remap = []
        for index, (key, value) in enumerate(signature_outputs.items()):
            full_signatures_outputs_remap.append(value)

# Generate random input data.
num_inputs = len(interpreter.get_input_details())
for i in range(num_inputs):
    input_details = interpreter.get_input_details()[i]
    if input_details["dtype"] == np.float32:
        input_data = np.array(np.random.random_sample(input_details["shape"]),
                              input_details["dtype"])
        input_dtype = "float32"
    elif input_details["dtype"] == np.uint8:
        input_data = np.array(np.random.randint(0, 256, size=input_details["shape"]),
                              input_details["dtype"])
        input_dtype = "uint8"
    elif input_details["dtype"] == np.int16:
        input_data = np.array(np.random.randint(0, 100, size=input_details["shape"]),
                              input_details["dtype"])
        input_dtype = "int16"
    elif input_details["dtype"] == np.int32:
        input_data = np.array(np.random.randint(0, 100, size=input_details["shape"]),
                              input_details["dtype"])
        input_dtype = "int32"
    elif input_details["dtype"] == np.int64:
        input_data = np.array(np.random.randint(0, 100, size=input_details["shape"]),
                              input_details["dtype"])
        input_dtype = "int64"
    elif input_details["dtype"] == np.bool_:
        input_data = np.array(
            np.random.choice(a=[True, False], size=input_details["shape"]),
            input_details["dtype"])
        input_dtype = "bool"
    else:
        raise SystemExit("Unsupported input dtype")

    interpreter.set_tensor(input_details["index"], input_data)
    input_data.tofile(circle_model + ".input" + str(i))
    input_details["shape"].tofile(circle_model + ".input" + str(i) + ".shape", sep=',')
    with open(circle_model + ".input" + str(i) + ".dtype", 'w') as dtype_file:
        dtype_file.write(input_dtype)

# Do inference
interpreter.invoke()

# Execute luci interpreter.
subprocess.run([
    driver, circle_model,
    str(num_inputs), circle_model + ".input", circle_model + ".output"
],
               check=True)

# Compare the results.
inpt_output_details = interpreter.get_output_details()
for idx in range(len(inpt_output_details)):
    output_details = inpt_output_details[idx]
    output_data = np.fromfile(circle_model + ".output" + str(idx),
                              output_details["dtype"])
    shape_file = open(circle_model + ".output" + str(idx) + ".shape", 'r')
    output_shape = [int(i) for i in shape_file.read().split(',')]
    luci_output_data = np.reshape(output_data, output_shape)
    output_tensor = output_details["index"]
    if full_signatures_outputs_remap != None:
        output_tensor = full_signatures_outputs_remap[idx]
    intp_output_data = interpreter.get_tensor(output_tensor)
    try:
        if output_details["dtype"] == np.uint8:
            if np.allclose(luci_output_data, intp_output_data, rtol=rtolint,
                           atol=atolint) == False:
                print("intp_output_data", intp_output_data)
                print("luci_output_data", luci_output_data)
                raise SystemExit("Execution result of " + tflite_model +
                                 " does not match with " + circle_model)
            output_dtype = "uint8"
        elif output_details["dtype"] == np.float32:
            if np.allclose(luci_output_data, intp_output_data, rtol=rtolf32,
                           atol=atolf32) == False:
                print("intp_output_data", intp_output_data)
                print("luci_output_data", luci_output_data)
                raise SystemExit("Execution result of " + tflite_model +
                                 " does not match with " + circle_model)
            output_dtype = "float32"
        elif output_details["dtype"] == np.int64:
            if np.allclose(luci_output_data, intp_output_data, rtol=rtolint,
                           atol=atolint) == False:
                print("intp_output_data", intp_output_data)
                print("luci_output_data", luci_output_data)
                raise SystemExit("Execution result of " + tflite_model +
                                 " does not match with " + circle_model)
            output_dtype = "int64"
        elif output_details["dtype"] == np.int32:
            if np.allclose(luci_output_data, intp_output_data, rtol=rtolint,
                           atol=atolint) == False:
                print("intp_output_data", intp_output_data)
                print("luci_output_data", luci_output_data)
                raise SystemExit("Execution result of " + tflite_model +
                                 " does not match with " + circle_model)
            output_dtype = "int32"
        elif output_details["dtype"] == np.int16:
            if np.allclose(luci_output_data, intp_output_data, rtol=rtolint,
                           atol=atolint) == False:
                print("intp_output_data", intp_output_data)
                print("luci_output_data", luci_output_data)
                raise SystemExit("Execution result of " + tflite_model +
                                 " does not match with " + circle_model)
            output_dtype = "int16"
        elif output_details["dtype"] == np.bool_:
            if np.allclose(luci_output_data, intp_output_data, rtol=0, atol=0) == False:
                print("intp_output_data", intp_output_data)
                print("luci_output_data", luci_output_data)
                raise SystemExit("Execution result of " + tflite_model +
                                 " does not match with " + circle_model)
            output_dtype = "bool"
        else:
            raise SystemExit("Unsupported data type: ", output_details["dtype"])

        # save outputN.dtype file
        with open(circle_model + ".output" + str(idx) + ".dtype", 'w') as dtype_file:
            dtype_file.write(output_dtype)
    except:
        print(traceback.format_exc())
        quit(255)

quit(0)
