#!/usr/bin/python3

import os
from os.path import dirname, basename, isdir, realpath, normpath, join
import argparse
from op_list_parser import OpListParser
import subprocess


class BackendProfiler():
    def __init__(self, args):
        self.nnpackage_dir = args.nnpackage_dir
        self.num_threads = args.num_threads
        self.script_path = realpath(__file__)
        self.root_path = dirname(dirname(dirname(self.script_path)))
        self.remote_base_dir = '/tmp/ONE'
        self.remote_host = f"{args.user}@{args.ip}" if args.user != None else args.ip
        self.remote_path = f"{self.remote_host}:{self.remote_base_dir}"

    def sync(self):
        bin_dir = join(self.root_path, "Product/armv7l-linux.release/out/")
        if (not isdir(bin_dir)):
            print(f"Build dir [{bin_dir}] is not exist")
            exit()
        elif (not isdir(self.nnpackage_dir)):
            print(f"nnpackage dir [{realpath(self.nnpackage_dir)}] is not exist")
            exit()
        else:
            # Syne ONE runtime
            subprocess.call([
                "rsync", "-azv", "--exclude", "test-suite.tar.gz", bin_dir,
                self.remote_path
            ])
            # Sync target nnpackage
            subprocess.call(["rsync", "-azv", self.nnpackage_dir, self.remote_path])

    def profile(self):
        backend_op_list = OpListParser().parse()
        backend_list = ["cpu"]
        backend_list.extend([backend for backend in backend_op_list])

        nnpkg_dir = basename(normpath(self.nnpackage_dir))
        remote_bin = join(self.remote_base_dir, "bin/nnpackage_run")
        remote_nnpkg = join(self.remote_base_dir, nnpkg_dir)

        for backend in backend_list:
            trace_name = f"{nnpkg_dir}_{backend}_{self.num_threads}"
            command = ["ssh", f"{self.remote_host}"]
            command.append(
                f"TRACE_FILEPATH={join(self.remote_base_dir,'traces',trace_name)}")
            for target_backend, op_list in backend_op_list.items():
                if backend == target_backend:
                    for op in op_list:
                        command.append(f"OP_BACKEND_{op}={backend}")
            command.append(f"EIGEN_THREADS={self.num_threads}")
            command.append(f"XNNPACK_THREADS={self.num_threads}")
            command.append(f"RUY_THREADS={self.num_threads}")
            command.append(f"BACKENDS=\'{';'.join(backend_list)}\'")
            command.append(f"OP_SEQ_MAX_NODE=1")
            command.append(f"{remote_bin}")
            command.append(f"--nnpackage")
            command.append(f"{remote_nnpkg}")
            command.append(f"-w5 -r50")
            print(command)
            print(' '.join(command))
            subprocess.call(command)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("nnpackage_dir", type=str, help="nnpackage folder to profile")
    arg_parser.add_argument("-n",
                            "--num_threads",
                            type=int,
                            default=1,
                            help="Number of threads used by one runtime")
    arg_parser.add_argument("--ip", type=str, help="IP address of remote client")
    arg_parser.add_argument("-u", "--user", type=str, help="User of remote client")
    args = arg_parser.parse_args()

    backend_profiler = BackendProfiler(args)
    backend_profiler.sync()
    backend_profiler.profile()
