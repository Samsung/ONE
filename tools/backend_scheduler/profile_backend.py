#!/usr/bin/python3

import os
from os.path import dirname, basename, isdir, realpath, normpath
import argparse


def parse_op_list():
    script_dir = dirname(realpath(__file__))
    print(script_dir)
    op_list_file = os.path.join(script_dir, "op_list.txt")
    backend_op_list = {}

    with open(op_list_file, 'r') as f:
        lines = f.readlines()

        for line in lines:
            line = line.rstrip()
            backend, _, op_list_str = line.partition(':')
            op_list = op_list_str.split(',')
            backend_op_list[backend] = op_list
    return backend_op_list


def main(args):
    script_path = realpath(__file__)
    root_path = dirname(dirname(dirname(script_path)))
    backend_op_list = parse_op_list()
    backend_list = ["cpu"]
    backend_list.extend([backend for backend in backend_op_list])
    os.chdir(root_path)

    if (isdir('./Product/armv7l-linux.release')):
        for backend in backend_list:
            trace_name = "{}_{}_{}_{}".format("armv7l",
                                              basename(normpath(args.nnpackage_dir)),
                                              backend, args.num_threads)
            command = "TRACE_FILEPATH={}/traces/{}".format(
                dirname(script_path), trace_name)
            for target_backend, op_list in backend_op_list.items():
                if backend == target_backend:
                    for op in op_list:
                        command += " OP_BACKEND_{}={}".format(op, backend)
            command += " EIGEN_THREADS={}".format(args.num_threads)
            command += " XNNPACK_THREADS={}".format(args.num_threads)
            command += " RUY_THREADS={}".format(args.num_threads)
            command += " BACKENDS='{}'".format(';'.join(backend_list))
            command += " OP_SEQ_MAX_NODE=1"
            command += " ./Product/armv7l-linux.release/out/bin/nnpackage_run"
            command += " --nnpackage {}".format(normpath(args.nnpackage_dir))
            command += " -w5 -r50"
            print(command)
            os.system(command)
    else:
        print("./Product/armv7l-linux.release not exist")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("nnpackage_dir", type=str, help="nnpackage folder to profile")
    arg_parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of threads used by one runtime")
    args = arg_parser.parse_args()

    main(args)
