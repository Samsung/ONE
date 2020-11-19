#!/usr/bin/python3

import os
import json
from os.path import dirname, basename, isdir, realpath
import argparse


def main(args):
    script_path = realpath(__file__)
    root_path = dirname(dirname(dirname(script_path)))
    os.chdir(root_path)

    op_time = {}
    for trace_file in os.listdir('./tools/backend_scheduler/traces'):
        if trace_file.endswith('.json') or trace_file.endswith('.md'):
            continue
        if args.nnpackage_name not in trace_file:
            continue
        with open('./tools/backend_scheduler/traces/' + trace_file, 'r') as f:
            data = json.load(f)
            execution_data = data['Execution_Data']
            for entry in execution_data:
                if entry == "memory" or entry == "runtime":
                    continue
                backend = entry
                backend_data = execution_data[backend]
                for op in backend_data:
                    op_index = op.split(' ')[0][1:]
                    op_type = op.split(' ')[1]
                    time = int(backend_data[op]["Avg_Time"])
                    if op_index not in op_time.keys():
                        op_time[op_index] = {backend: time}
                        op_time[op_index].update({"type": op_type})
                    else:
                        op_time[op_index].update({backend: time})
            print(op_time)

    backend_mapping = {}
    for op_index, value in op_time.items():
        op_type = value['type']
        if op_type != 'Conv2D':
            continue
        cpu_time = value['cpu']
        ruy_time = value['ruy']

        if cpu_time < ruy_time:
            backend_mapping[op_index] = 'cpu'
            print("Use cpu : " + op_index)
        else:
            backend_mapping[op_index] = 'ruy'

    backend_conf = ""
    for op_index, backend in backend_mapping.items():
        backend_conf += "{}={};".format(op_index, backend)

    print(backend_conf)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("nnpackage_name",
                            type=str,
                            help="nnpackage folder to profile")
    args = arg_parser.parse_args()

    main(args)
