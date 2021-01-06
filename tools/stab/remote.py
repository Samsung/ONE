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

import subprocess, logging
from pathlib import Path


class RemoteSSH():
    """
    Execute commands on remove device using SSH
    """
    def __init__(self, user, ip, nnpkg_dir, num_threads):
        self.base_dir = Path('/tmp/ONE')
        self.trace_dir = 'traces'
        self.host = f"{user}@{ip}" if user != None else ip
        self.nnpkg_dir = Path(nnpkg_dir).resolve()
        self.root_path = Path(__file__).resolve().parents[2]
        self.num_threads = num_threads

    def sync_binary(self):
        bin_dir = self.root_path / 'Product/armv7l-linux.release/out'
        if (not bin_dir.is_dir()):
            logging.warn(f"Build dir [{bin_dir}] is not exist")
            exit()
        elif (not self.nnpkg_dir.is_dir()):
            logging.warn(f"nnpackage dir [{self.nnpkg_dir}] is not exist")
            exit()
        else:
            # Syne ONE runtime
            subprocess.call([
                "rsync", "-az", "--exclude", "test-suite.tar.gz", bin_dir,
                self.remote(self.base_dir)
            ])
            # Sync target nnpackage
            subprocess.call(["rsync", "-az", self.nnpkg_dir, self.remote(self.base_dir)])

    def sync_trace(self, backend):
        remote_trace_path = self.remote_trace_path(backend)
        local_trace_path = self.local_trace_path(backend)
        local_trace_path.parent.mkdir(parents=True, exist_ok=True)
        logging.debug(f"Remote trace path : {self.remote(remote_trace_path)}")
        logging.debug(f"Local trace path : {local_trace_path}")
        # Sync trace file
        subprocess.call(
            ["rsync", "-az",
             self.remote(remote_trace_path), local_trace_path])

    def profile_backend(self, backend, backend_op_list):
        nnpkg_run_path = self.base_dir / 'bin/nnpackage_run'
        nnpkg_path = self.base_dir / self.nnpkg_dir.name

        cmd = ["ssh", f"{self.host}"]
        cmd += [f"TRACE_FILEPATH={self.remote_trace_path(backend)}"]
        for target_backend, op_list in backend_op_list.items():
            if backend == target_backend:
                for op in op_list:
                    cmd += [f"OP_BACKEND_{op}={backend}"]
        cmd += [f"EIGEN_THREADS={self.num_threads}"]
        cmd += [f"XNNPACK_THREADS={self.num_threads}"]
        cmd += [f"RUY_THREADS={self.num_threads}"]
        cmd += [f"BACKENDS=\'{';'.join(['cpu', backend])}\'"]
        cmd += [f"OP_SEQ_MAX_NODE=1"]
        cmd += [f"{nnpkg_run_path}"]
        cmd += [f"--nnpackage"]
        cmd += [f"{nnpkg_path}"]
        cmd += [f"-w5 -r50"]
        logging.debug(f"SSH command : {' '.join(cmd)}")
        subprocess.call(cmd)

    def base_path():
        pass

    def remote(self, path):
        return f"{self.host}:{path}"

    # TODO Create class for path generation
    def trace_name(self, backend):
        return f"{backend}_{self.num_threads}"

    def remote_trace_path(self, backend):
        return self.base_dir / self.trace_dir / self.trace_name(backend)

    def local_trace_path(self, backend):
        return Path(__file__).parent / self.trace_dir / self.trace_name(backend)
