#!/usr/bin/env python3

#
# generate_datafile.py
# - Generate Model input and expected datas for onert_train
#

import argparse
import os
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate Model input and expected datas for onert_train')
    parser.add_argument('tool')
    parser.add_argument('model')
    parser.add_argument('--input_name', '-i', default='input')
    parser.add_argument('--output_name', '-o', default='output')
    parser.add_argument('--num_runs', '-n', default=1, type=int)

    args = parser.parse_args()

    if os.path.exists(args.tool) == False:
        print("Tool file not found: " + args.tool)
        exit(1)

    if os.path.exists(args.model) == False:
        print("Model file not found: " + args.model)
        exit(1)

    base_name = os.path.splitext(args.model)[0]
    input_file = base_name + "." + args.input_name + '.bin'
    output_file = base_name + "." + args.output_name + '.bin'

    inf = open(input_file, 'wb')
    outf = open(output_file, 'wb')

    for idx in range(args.num_runs):
        input_idx_file = base_name + "." + args.input_name
        output_idx_file = base_name + "." + args.output_name

        subprocess.run([
            args.tool, '--modelfile', args.model, '--dump_input:raw', input_idx_file,
            '--dump:raw', output_idx_file
        ])

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

    print("Generated input and output files")
    print("Done")
    exit(0)
