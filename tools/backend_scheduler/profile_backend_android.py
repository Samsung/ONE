#!/usr/bin/python3

import os
from os.path import dirname, basename, isdir, realpath
import argparse


def main(args):
    script_path = realpath(__file__)
    root_path = dirname(dirname(dirname(script_path)))
    os.chdir(root_path)

    # kernel_conf_list = []
    # with open("backend_conf.txt") as f:
    #     kernel_conf_list = [line.strip() for line in f.readlines()]
    # print(kernel_conf_list)

    backend_list = ["cpu", "ruy"]

    if (isdir('./Product/aarch64-android.release')):
        # for index, kernel_conf in enumerate(kernel_conf_list):
        for index, backend in enumerate(backend_list):
            trace_name = "{}_{}".format(basename(dirname(args.nnpackage_dir)), str(index))
            command = "adb shell \"TRACE_FILEPATH=/data/local/tmp/traces/{}".format(
                trace_name)
            command += " OP_BACKEND_Conv2D={}".format(backend)
            command += " BACKENDS='cpu;ruy'"
            command += " OP_SEQ_MAX_NODE=1"
            command += " LD_LIBRARY_PATH=/data/local/tmp/nnfw_alpha/lib"
            command += " ./data/local/tmp/nnfw_alpha/nnpackage_run"
            command += " --nnpackage /data/local/tmp/{}".format(
                dirname(args.nnpackage_dir))
            command += " -w5 -r50\""
            print(command)
            os.system(command)
    else:
        print("./Product/armv7l-linux.release not exist")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("nnpackage_dir", type=str, help="nnpackage folder to profile")
    args = arg_parser.parse_args()

    main(args)
