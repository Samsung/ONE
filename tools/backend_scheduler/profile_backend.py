#!/usr/bin/python3

import os
from os.path import dirname, basename, isdir, realpath, normpath
import argparse


def main(args):
    script_path = realpath(__file__)
    root_path = dirname(dirname(dirname(script_path)))
    os.chdir(root_path)

    backend_list = ["cpu", "ruy"]

    if (isdir('./Product/armv7l-linux.release')):
        for index, backend in enumerate(backend_list):
            trace_name = "{}_{}_{}".format("armv7l", basename(
                normpath(args.nnpackage_dir)), backend)
            command = "TRACE_FILEPATH={}/traces/{}".format(
                dirname(script_path), trace_name)
            command += " OP_BACKEND_Conv2D={}".format(backend)
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
    args = arg_parser.parse_args()

    main(args)
