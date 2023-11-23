#!/usr/bin/env python3

#
# random_data_generator.py
# - Create random data files for the `onert_train` tool
#

import argparse
import os
import subprocess
import contextlib


def initParse():
    parser = argparse.ArgumentParser(
        description='Create random data files for onert_train')
    parser.add_argument('tool')
    parser.add_argument('model')
    parser.add_argument('--input-name', '-i', default='input')
    parser.add_argument('--output-name', '-o', default='output')
    parser.add_argument('--data-length', '-l', default=1, type=int)

    return parser.parse_args()


def generateData(onert_run_path, model, input_name, output_name, data_length):
    base_name = os.path.splitext(model)[0]
    input_file = base_name + "." + input_name + '.bin'
    output_file = base_name + "." + output_name + '.bin'

    inf = open(input_file, 'wb')
    outf = open(output_file, 'wb')

    for _ in range(data_length):
        input_idx_file = base_name + "." + input_name
        output_idx_file = base_name + "." + output_name

        subprocess.run(
            [
                onert_run_path, '--modelfile', model, '--dump_input:raw', input_idx_file,
                '--dump:raw', output_idx_file
            ],
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL)

        input_idx_file += '.0'
        output_idx_file += '.0'

        if os.path.exists(input_idx_file) == False or os.path.exists(
                output_idx_file) == False:
            print("Failed to generate nth input and output files")
            exit(1)

        with open(input_idx_file, 'rb') as f:
            inf.write(f.read())
        with open(output_idx_file, 'rb') as f:
            outf.write(f.read())

        with contextlib.suppress(FileNotFoundError):
            os.remove(input_idx_file)
            os.remove(output_idx_file)

    print("Generated input and output files")
    print("Done")


if __name__ == "__main__":

    args = initParse()

    if os.path.exists(args.tool) == False:
        print("Tool file not found: " + args.tool)
        exit(1)

    if os.path.exists(args.model) == False:
        print("Model file not found: " + args.model)
        exit(1)

    generateData(args.tool, args.model, args.input_name, args.output_name,
                 args.data_length)
    exit(0)
