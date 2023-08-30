#!/usr/bin/env python3
import argparse
import os

from test_utils import TestCase
from test_utils import TestRunner
from test_utils import gen_random_tensor


class Conv2D_000_Q8(TestCase):
    def __init__(self):
        self.name = "Conv2D_000_Q8"

    def generate(self) -> dict:
        json_content = dict()

        # Generate ifm
        json_content['ifm'] = gen_random_tensor(
            "uint8",  # dtype_str
            (1),  # scale_shape
            (1),  # zerop_shape
            0)  # quantized_dimension

        # Generate ker
        json_content['ker'] = gen_random_tensor(
            "uint8",  #dtype_str
            (1),  # scale_shape
            (1),  # zerop_shape
            0,  # quantized_dimension
            (1, 1, 1, 2))  # value_shape (OHWI)

        # Generate bias
        json_content['bias'] = gen_random_tensor(
            "int32",  #dtype_str
            (1),  # scale_shape
            (1),  # zerop_shape
            0,  # quantized_dimension
            (1))  # value_shape

        # Generate ofm
        json_content['ofm'] = gen_random_tensor(
            "uint8",  # dtype_str
            (1),  # scale_shape
            (1),  # zerop_shape
            0)  # quantized_dimension

        return json_content


#
# This script generates a pack of random input data (.h5) expected by the input circle model
#
# Basic usage:
#   gen_test_data.py --output_dir <path/to/directory>
#   ex: gen_test_data.py
#   (This will create qparam.json and corresponding numpy files under <path/to/directory>)
#
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, required=True)
args = parser.parse_args()

output_dir = args.output_dir

test_runner = TestRunner(output_dir)

test_runner.register(Conv2D_000_Q8())

test_runner.run()
