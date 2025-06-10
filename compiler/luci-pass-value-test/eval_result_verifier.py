#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import subprocess
import argparse
import traceback

#
# This script was copied from luci-value-test with input arguments are tflite and circle path
#
parser = argparse.ArgumentParser()
parser.add_argument('--driver', type=str, required=True)
parser.add_argument('--tflite', type=str, required=True)
parser.add_argument('--circle', type=str, required=True)
args = parser.parse_args()

driver = args.driver
tflite_model = args.tflite
circle_model = args.circle

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
    elif input_details["dtype"] == np.uint8:
        input_data = np.array(np.random.randint(0, 256, size=input_details["shape"]),
                              input_details["dtype"])
    elif input_details["dtype"] == np.int16:
        input_data = np.array(np.random.randint(0, 100, size=input_details["shape"]),
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
            if np.allclose(luci_output_data, intp_output_data, rtol=0, atol=0) == False:
                raise SystemExit("Execution result of " + tflite_model +
                                 " does not match with " + circle_model)
        elif output_details["dtype"] == np.float32:
            if np.allclose(luci_output_data, intp_output_data, rtol=1.e-5,
                           atol=1.e-5) == False:
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
        elif output_details["dtype"] == np.int16:
            if np.allclose(luci_output_data, intp_output_data, rtol=0, atol=0) == False:
                raise SystemExit("Execution result of " + tflite_model +
                                 " does not match with " + circle_model)
        else:
            raise SystemExit("Unsupported data type: ", output_details["dtype"])
    except:
        print(traceback.format_exc())
        quit(255)

quit(0)
