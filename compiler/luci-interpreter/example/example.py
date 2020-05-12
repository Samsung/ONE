#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import subprocess
import argparse

# Basic usage:
#   example.py --driver build/compiler/luci-interpreter/example/luci_interpreter_example
#              --model inception_v3.tflite
parser = argparse.ArgumentParser()
parser.add_argument('--driver', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
args = parser.parse_args()

driver = args.driver
model = args.model

# Build TensorFlow Lite interpreter.
interpreter = tf.lite.Interpreter(model)
interpreter.allocate_tensors()

# Generate input data.
assert len(interpreter.get_input_details()) == 1
input_details = interpreter.get_input_details()[0]
input_data = np.array(
    np.random.random_sample(input_details["shape"]), input_details["dtype"])
interpreter.set_tensor(input_details["index"], input_data)
interpreter.set_tensor(input_details["index"], input_data)

interpreter.invoke()

# Get reference output data.
assert len(interpreter.get_output_details()) == 1
output_details = interpreter.get_output_details()[0]
ref_output_data = interpreter.get_tensor(output_details["index"])

# Execute luci interpreter.
input_data.tofile("input.dat")
subprocess.run([driver, model, "input.dat", "output.dat"])
output_data = np.fromfile("output.dat", output_details["dtype"])

# Compare the results.
if output_details["dtype"] == np.uint8:
    # Ideally, the match should be exact.
    assert np.allclose(output_data, ref_output_data, rtol=0, atol=1)
else:
    assert np.allclose(output_data, ref_output_data, rtol=1.e-5, atol=1.e-5)

print("SUCCESS")
