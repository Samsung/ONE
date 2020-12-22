#!/usr/bin/python3

import argparse
from op_list_parser import OpListParser
from remote import RemoteSSH


class BackendProfiler():
    def __init__(self, user, ip, nnpackage_dir, num_threads):
        self.remote_ssh = RemoteSSH(user, ip, nnpackage_dir, num_threads)
        self.backend_op_list = OpListParser().parse()
        self.backend_list = ["cpu"]
        self.backend_list.extend([backend for backend in self.backend_op_list])

    def sync(self):
        self.remote_ssh.sync_binary()

    def profile(self):
        for backend in self.backend_list:
            self.remote_ssh.profile_backend(backend, self.backend_op_list)
            self.remote_ssh.sync_trace(backend)


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

    backend_profiler = BackendProfiler(args.user, args.ip, args.nnpackage_dir,
                                       args.num_threads)
    backend_profiler.sync()
    backend_profiler.profile()
