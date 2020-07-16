#!/usr/bin/env python3
import h5py as h5
import numpy as np
import argparse
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "-l",
    "--data_list",
    type=str,
    help=
    "Path to the text file which lists the absolute paths of the raw image data files to be converted.",
    required=True)
parser.add_argument(
    "-o", "--output_path", type=str, help="Path to the output hdf5 file.", required=True)

args = parser.parse_args()
data_list = args.data_list
output_path = args.output_path

# Create h5 file
h5_file = h5.File(output_path, 'w')
group = h5_file.create_group("value")
# We assume the raw input data have the correct type/shape for the corresponding model
# If this flag is set in the hdf5 file, record-minmax will skip type/shape check
group.attrs['rawData'] = '1'

# Data list
datalist = []
with open(data_list, 'r') as f:
    lines = f.readlines()
    for line in lines:
        datalist.append(line.rstrip())

# Input files
num_converted = 0
for imgdata in datalist:
    with open(imgdata, 'r') as f:
        sample = group.create_group(str(num_converted))
        num_converted += 1
        filename = os.path.basename(imgdata)
        sample.attrs['desc'] = filename
        raw_data = bytearray(f.read())
        # The target model is DNN for handling an input image
        sample.create_dataset('0', data=raw_data)

h5_file.close()

print("Raw image data have been packaged to " + output_path)
print("Number of packaged data: " + str(num_converted))
