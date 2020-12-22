#!/usr/bin/python3

import subprocess
from os.path import dirname, basename, isdir, realpath, normpath, join


class RemoteSSH():
    def __init__(self, user, ip, nnpkg_dir, num_threads):
        self.base_dir = '/tmp/ONE'
        self.trace_dir = 'traces/'
        self.host = f"{user}@{ip}" if user != None else ip
        self.nnpkg_dir = nnpkg_dir
        self.root_path = dirname(dirname(dirname(realpath(__file__))))
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
                "rsync", "-az", "--exclude", "test-suite.tar.gz", bin_dir,
                self.remote(self.base_dir)
            ])
            # Sync target nnpackage
            subprocess.call(["rsync", "-az", self.nnpkg_dir, self.remote(self.base_dir)])

    def sync_trace(self, backend):
        remote_trace_path = self.remote_trace_path(backend)
        local_trace_path = self.local_trace_path(backend)
        # Sync trace file
        subprocess.call(
            ["rsync", "-az",
             self.remote(remote_trace_path), local_trace_path])

    def profile_backend(self, backend, backend_op_list):
        nnpkg_run_path = join(self.base_dir, "bin/nnpackage_run")
        nnpkg_dir = basename(normpath(self.nnpkg_dir))
        nnpkg_path = join(self.base_dir, nnpkg_dir)

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
        print(' '.join(cmd))
        subprocess.call(cmd)

    def base_path():
        pass

    def remote(self, path):
        return f"{self.host}:{path}"

    # TODO Create class for path generation
    def trace_file(self, backend):
        nnpkg_dir = basename(normpath(self.nnpkg_dir))
        return f"{nnpkg_dir}_{backend}_{self.num_threads}"

    def remote_trace_path(self, backend):
        return f"{join(self.base_dir,self.trace_dir,self.trace_file(backend))}"

    def local_trace_path(self, backend):
        return join(self.root_path, self.trace_dir, f"{backend}_{self.num_threads}")
