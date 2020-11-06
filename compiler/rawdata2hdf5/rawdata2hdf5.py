#!/usr/bin/env python3

# Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    "Path to the text file which lists the absolute paths of the raw data files to be converted.",
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

if os.path.isfile(data_list) == False:
    raise SystemExit("No such file. " + data_list)

# Data list
datalist = []
with open(data_list, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if line.strip():
            filename = line.rstrip()
            if os.path.isfile(filename):
                datalist.append(filename)
            else:
                raise SystemExit("No such file. " + filename)

# Input files
num_converted = 0
for rawdata in datalist:
    with open(rawdata, 'rb') as f:
        sample = group.create_group(str(num_converted))
        num_converted += 1
        filename = os.path.basename(rawdata)
        sample.attrs['desc'] = filename
        raw_data = bytearray(f.read())
        # The target model is DNN for handling an input data
        sample.create_dataset('0', data=raw_data)

h5_file.close()

print("Raw data have been packaged to " + output_path)
print("Number of packaged data: " + str(num_converted))
