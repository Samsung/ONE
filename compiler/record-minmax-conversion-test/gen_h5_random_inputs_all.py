#!/usr/bin/env python3
import h5py as h5
import numpy as np
import tensorflow as tf
import argparse
import os

#
# This script generates a pack of random input data (.h5) expected by the input tflite model
#
parser = argparse.ArgumentParser()
parser.add_argument('--num_data', type=int, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--artifact_dir', type=str, required=True)
parser.add_argument('--model', type=str, required=True, nargs='+')
args = parser.parse_args()

num_data = args.num_data
output_dir = args.output_dir
artifact_dir = args.artifact_dir
model_list = args.model

for model_name in model_list:
    model_path = os.path.join(artifact_dir, model_name + '.tflite')
    h5_path = os.path.join(output_dir, model_name + '.tflite.input.h5')
    # Build TFLite interpreter. (to get the information of model input)
    interpreter = tf.lite.Interpreter(model_path)
    input_details = interpreter.get_input_details()

    # Create h5 file
    h5_file = h5.File(h5_path, 'w')
    group = h5_file.create_group("value")
    group.attrs['desc'] = "Input data for " + model_path

    # Generate random data
    for i in range(num_data):
        sample = group.create_group(str(i))
        sample.attrs['desc'] = "Input data " + str(i)

        for j in range(len(input_details)):
            input_detail = input_details[j]
            print(input_detail["dtype"])
            if input_detail["dtype"] == np.bool_:
                # Generate random bool [0, 1]
                input_data = np.array(
                    np.random.random_integers(0, 1, input_detail["shape"]),
                    input_detail["dtype"])
            elif input_detail["dtype"] == np.float32:
                # Generate random input [-5, 5)
                input_data = np.array(
                    10 * np.random.random_sample(input_detail["shape"]) - 5,
                    input_detail["dtype"])
            sample.create_dataset(str(j), data=input_data)

    h5_file.close()
