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
    #  backend_list = ["cpu", "ruy"]

    op_time = {}
    for trace_file in os.listdir('./tools/backend_scheduler/traces'):
        if trace_file.endswith('.json') or trace_file.endswith('.md'):
            continue
        trace_name = "_".join(trace_file.split("_")[0:-2])
        num_threads = trace_file.split("_")[-1]
        if args.trace_name not in trace_name:
            continue
        if num_threads != str(args.num_threads):
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
                    time = int(backend_data[op]["Min_Time"])
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
    for op_index, value in sorted(op_time.items()):
        op_type = value['type']
        if op_type != 'Conv2D' and op_type != 'FullyConnected':
            continue

        print("----- Operation {} -----".format(op_index))
        op_infer_time = 0
        for backend in backend_list:
            backend_time = value[backend]

            print("{}[{}]".format(backend, backend_time))

            if op_infer_time == 0 or backend_time < op_infer_time:
                op_infer_time = backend_time
                backend_mapping[op_index] = backend

    # Count backends
    for op_index, backend in backend_mapping.items():
        backend_count[backend] += 1

    # Find default backend for Conv2D
    default_backend = max(backend_count, key=backend_count.get)

    # Create OP_BACKEND_MAP string
    backend_conf = ""
    for op_index, backend in sorted(backend_mapping.items()):
        if backend != default_backend:
            backend_conf += "{}={};".format(op_index, backend)

    print("-------- Expected inference time ---------")
    single_backend_time = 0
    schedule_time = 0
    for op_index, value in sorted(op_time.items()):
        op_type = value['type']
        if op_type != 'Conv2D' and op_type != 'FullyConnected':
            single_backend_time += value["cpu"]
            schedule_time += value["cpu"]
            continue
        else:
            op_backend = backend_mapping[op_index]
            single_backend_time += value[default_backend]
            schedule_time += value[op_backend]
            if (default_backend != op_backend):
                print("[{}] {} -> {} : {:.2f} ms decrease".format(
                    op_index, default_backend, op_backend,
                    (value[default_backend] - value[op_backend]) / 1000))

    print("{} backend : {:.2f} ms".format(default_backend, single_backend_time / 1000))
    print("Mixed backend : {:.2f} ms".format(schedule_time / 1000))

    print("-------- Backend Scheduling --------")
    print(
        "OP_BACKEND_MAP='{}' OP_BACKEND_Conv2D={} BACKENDS='{}' EIGEN_THREADS={} RUY_THREADS={} XNNPACK_THREADS={}".
        format(backend_conf, default_backend, ';'.join(backend_list), args.num_threads,
               args.num_threads, args.num_threads))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("trace_name", type=str, help="Recorded trace file name")
    arg_parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of threads used by one runtime")
    args = arg_parser.parse_args()

    main(args)
