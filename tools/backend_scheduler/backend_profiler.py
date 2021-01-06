#!/usr/bin/env python3

# Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse, logging
from op_list_parser import OpListParser
from remote import RemoteSSH


class BackendProfiler():
    """
    Run ONE runtime on remote device to create TRACE file which has operation execution time

    TODO : Support Android device profiling
    """
    def __init__(self, user, ip, nnpackage_dir, num_threads):
        self.remote_ssh = RemoteSSH(user, ip, nnpackage_dir, num_threads)
        self.backend_op_list = OpListParser().parse()
        self.backend_list = ["cpu"]
        self.backend_list.extend([backend for backend in self.backend_op_list])

    def sync(self):
        logging.info("Upload ONE runtime and nnpackage to remote device")
        self.remote_ssh.sync_binary()

    def profile(self):
        for backend in self.backend_list:
            logging.info(f"Profiling {backend} backend")
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
