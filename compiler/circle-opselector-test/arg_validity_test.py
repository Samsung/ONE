#!/usr/bin/env python3
import os
import argparse

# This script tests circle-opselector's option validity.
parser = argparse.ArgumentParser()
parser.add_argument('--opselector', type=str, required=True)
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()
opselector = args.opselector
circle_input = args.input
circle_output = args.output


def run_opselector(opselector, option, circle_input, circle_output):
    """
    Run CircleOpselector
    """
    print('---', f'{opselector} {option} {circle_input} {circle_output}')
    result = os.system(f'{opselector} {option} {circle_input} {circle_output}')
    return result


options = {
    # by_id
    '--by_id "1,2"': 0,
    '--by_id "1, 2"': 0,
    '--by_id "1-2"': 0,
    '--by_id "3, 1"': 0,
    '--by_id "1 - 2"': 0,
    '--by_id "0, 0, 1"': 0,  # duplicaged nodes -> 0, 1
    '\"0-3\"': 256,  # no by_id or by_name
    '--by_id "a,b"': 256,  # not integer 
    '--by_id "0.1"': 256,  # not integer 
    '--by_id "0--1"': 256,  # double hyphen
    '--by_id "-1-3"': 256,  # not positive integer
}

failed = 0
for option, value in options.items():
    result = run_opselector(opselector, option, circle_input, circle_output)
    if result != value:
        print(
            f'fail, circle_input: {circle_input} option: {option}, expected: {value}, result: {result}'
        )
        failed = 1

if failed:
    quit(255)
else:
    quit(0)
