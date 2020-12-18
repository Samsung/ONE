#!/usr/bin/python3

import subprocess
from os.path import dirname, basename, isdir, realpath, normpath, join


class RemoteSSH():
    def __init__(self, user, ip, nnpkg_dir, num_threads):
        self.base_dir = '/tmp/ONE'
        self.host = f"{user}@{ip}" if user != None else ip
        self.nnpkg_dir = nnpkg_dir
        self.root_path = dirname(dirname(dirname(realpath(__file__))))
        self.base_path = f"{self.host}:{self.base_dir}"
        self.num_threads = num_threads

    def sync_binary(self):
        bin_dir = join(self.root_path, "Product/armv7l-linux.release/out/")
        if (not isdir(bin_dir)):
            print(f"Build dir [{bin_dir}] is not exist")
            exit()
        elif (not isdir(self.nnpkg_dir)):
            print(f"nnpackage dir [{realpath(self.nnpkg_dir)}] is not exist")
            exit()
        else:
            # Syne ONE runtime
            subprocess.call([
                "rsync", "-azv", "--exclude", "test-suite.tar.gz", bin_dir, self.base_path
            ])
            # Sync target nnpackage
            subprocess.call(["rsync", "-azv", self.nnpkg_dir, self.base_path])

    def sync_trace():
        pass

    def profile_backend(self, backend, backend_op_list):
        nnpkg_dir = basename(normpath(self.nnpkg_dir))
        nnpkg_run = join(self.base_dir, "bin/nnpackage_run")
        nnpkg = join(self.base_dir, nnpkg_dir)

        trace_name = f"{nnpkg_dir}_{backend}_{self.num_threads}"
        command = ["ssh", f"{self.host}"]
        command.append(f"TRACE_FILEPATH={join(self.base_dir,'traces',trace_name)}")
        for target_backend, op_list in backend_op_list.items():
            if backend == target_backend:
                for op in op_list:
                    command.append(f"OP_BACKEND_{op}={backend}")
        command.append(f"EIGEN_THREADS={self.num_threads}")
        command.append(f"XNNPACK_THREADS={self.num_threads}")
        command.append(f"RUY_THREADS={self.num_threads}")
        command.append(f"BACKENDS=\'{';'.join(['cpu', backend])}\'")
        command.append(f"OP_SEQ_MAX_NODE=1")
        command.append(f"{nnpkg_run}")
        command.append(f"--nnpackage")
        command.append(f"{nnpkg}")
        command.append(f"-w5 -r50")
        print(command)
        print(' '.join(command))
        subprocess.call(command)

    def base_path():
        pass

    def trace_path():
        pass
