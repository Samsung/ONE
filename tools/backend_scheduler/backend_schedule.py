#!/usr/bin/python3

import os
import json
import argparse
from pathlib import Path
from op_list_parser import OpListParser
from nnpkg_helper import NnpkgHelper


class BackendScheduler:
    def __init__(self, nnpkg_dir, num_threads):
        self.nnpkg_dir = Path(nnpkg_dir).resolve()
        self.num_threads = num_threads
        self.root_path = Path(__file__).resolve().parents[2]
        self.nnpkg_helper = NnpkgHelper()

    def read_traces(self, backend_list):
        op_time = {}
        inference_time = {}
        for backend in backend_list:
            try:
                with open(f"./traces/{backend}_{self.num_threads}") as f:
                    data = json.load(f)
                    execution_data = data['Execution_Data']
                    for entry in execution_data:
                        if entry == "memory":
                            continue
                        elif entry == "runtime":
                            inference_time[backend] = execution_data['runtime']['Graph'][
                                'Avg_Time']
                            continue
                        op_backend = entry
                        backend_data = execution_data[op_backend]
                        for op in backend_data:
                            op_index = int(op.split(' ')[0][1:])
                            op_type = op.split(' ')[1]
                            time = int(backend_data[op]["Avg_Time"])
                            if op_index not in op_time.keys():
                                op_time[op_index] = {op_backend: time}
                                op_time[op_index].update({"type": op_type})
                            else:
                                op_time[op_index].update({op_backend: time})
            except IOError as e:
                print(e)
        return op_time, inference_time

    def schedule(self):
        backend_op_list = OpListParser().parse()
        backend_list = ["cpu"]
        backend_list.extend([backend for backend in backend_op_list])
        os.chdir(self.root_path)

        op_time, backend_infer_time = self.read_traces(backend_list)

        backend_mapping = {}

        target_ops = set()
        for _, v in backend_op_list.items():
            target_ops.update(v)

        # Find fastest backend for each operation
        for op_index, value in sorted(op_time.items()):
            op_type = value['type']
            if op_type not in target_ops:
                continue

            # print("----- Operation {} -----".format(op_index))
            op_infer_time = 0
            for backend in backend_list:
                if backend not in value:
                    continue
                backend_time = value[backend]

                # print("{}[{}]".format(backend, backend_time))

                if op_infer_time == 0 or backend_time < op_infer_time:
                    op_infer_time = backend_time
                    backend_mapping[op_index] = backend

        # Find default backend for Conv2D
        default_backend = min(backend_infer_time, key=backend_infer_time.get)

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
            if op_type not in target_ops:
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

        for backend in backend_list:
            print(f"{backend} backend : {backend_infer_time[backend]/1000:.2f} ms")
        print(f"Mixed backend : {schedule_time / 1000:.2f} ms")

        print("-------- Backend Scheduling --------")
        cmd = []
        cmd += [f"OP_BACKEND_MAP={backend_conf}"]
        for target_backend, op_list in backend_op_list.items():
            if default_backend == target_backend:
                for op in op_list:
                    cmd += [f"OP_BACKEND_{op}={default_backend}"]
        cmd += [f"BACKENDS={';'.join(backend_list)}"]
        cmd += [f"EIGEN_THREADS={self.num_threads}"]
        cmd += [f"RUY_THREADS={self.num_threads}"]
        cmd += [f"XNNPACK_THREADS={self.num_threads}"]
        print(' '.join(cmd))

        dst_dir = Path(__file__).parent / 'nnpkg_sched' / self.nnpkg_dir.name
        self.nnpkg_helper.copy(self.nnpkg_dir, dst_dir)
        self.nnpkg_helper.add_config(dst_dir, cmd)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("nnpackage_dir", type=str, help="nnpackage folder to profile")
    arg_parser.add_argument("--num_threads",
                            type=int,
                            default=1,
                            help="Number of threads used by one runtime")
    args = arg_parser.parse_args()

    backend_scheduler = BackendScheduler(args.nnpackage_dir, args.num_threads)
    backend_scheduler.schedule()
