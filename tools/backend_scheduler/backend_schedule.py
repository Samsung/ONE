#!/usr/bin/python3

import os
import json
from os.path import dirname, basename, isdir, realpath
import argparse


def main(args):
    script_path = realpath(__file__)
    root_path = dirname(dirname(dirname(script_path)))
    os.chdir(root_path)

    backend_list = ["cpu", "ruy", "xnnpack"]
    # backend_list = ["cpu", "ruy"]

    op_time = {}
    for trace_file in os.listdir('./tools/backend_scheduler/traces'):
        if trace_file.endswith('.json') or trace_file.endswith('.md'):
            continue
        if args.trace_name not in trace_file.rpartition('_')[0]:
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
                    op_index = int(op.split(' ')[0][1:])
                    op_type = op.split(' ')[1]
                    time = int(backend_data[op]["Avg_Time"])
                    if op_index not in op_time.keys():
                        op_time[op_index] = {backend: time}
                        op_time[op_index].update({"type": op_type})
                    else:
                        op_time[op_index].update({backend: time})

    backend_mapping = {}
    backend_count = {}
    for backend in backend_list:
        backend_count[backend] = 0

    # Find fastest library for each operation
    for op_index, value in op_time.items():
        op_type = value['type']
        if op_type != 'Conv2D':
            continue

        print("----- Operation {} -----".format(op_index))
        op_infer_time = 0
        for backend in backend_list:
            backend_time = value[backend]

            print("{}[{}]".format(backend, backend_time))

            if op_infer_time == 0 or backend_time < op_infer_time:
                # if op_index in backend_mapping:
                # print("{}[{}] < {}[{}]".format(backend_mapping[op_index], op_infer_time, backend, backend_time))

                op_infer_time = backend_time
                backend_mapping[op_index] = backend

    # Count backends
    for op_index, backend in backend_mapping.items():
        backend_count[backend] += 1

    # Find default backend for Conv2D
    default_backend = max(backend_count, key=backend_count.get)

    backend_conf = ""
    for op_index, backend in sorted(backend_mapping.items()):
        if backend != default_backend:
            backend_conf += "{}={};".format(op_index, backend)

    print("-------- Backend Scheduling --------")
    print("OP_BACKEND_MAP='{}' OP_BACKEND_Conv2D={} BACKENDS='{}'".format(
        backend_conf, default_backend, ';'.join(backend_list)))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("trace_name", type=str, help="Recorded trace file name")
    args = arg_parser.parse_args()

    main(args)
