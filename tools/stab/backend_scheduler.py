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

import json, logging
from pathlib import Path
from op_list_parser import OpListParser
from nnpkg_helper import NnpkgHelper


class BackendScheduler:
    """
    Read profiled data and select proper backend for each operation
    Scheduled nnpackage is saved at ./tools/stab/nnpkg_sched

    TODO : Use permutation time for better scheduling
    """
    def __init__(self, nnpkg_dir, num_threads):
        self.nnpkg_dir = Path(nnpkg_dir).resolve()
        self.num_threads = num_threads
        self.root_path = Path(__file__).parents[2]
        self.nnpkg_helper = NnpkgHelper()

    def read_traces(self, backend_list):
        op_time = {}
        inference_time = {}
        for backend in backend_list:
            try:
                # Trace file is located at ./tools/stab/traces
                trace_path = Path(
                    __file__).parent / 'traces' / f"{backend}_{self.num_threads}"
                logging.debug(f"Trace path : {trace_path}")
                with open(trace_path) as f:
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
                logging.warning(e)
        return op_time, inference_time

    def schedule(self):
        backend_op_list = OpListParser().parse()
        backend_list = ["cpu"]
        backend_list.extend([backend for backend in backend_op_list])

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

            logging.debug(f"----- Operation {op_index} -----")
            op_infer_time = 0
            for backend in backend_list:
                if backend not in value:
                    continue
                backend_time = value[backend]

                logging.debug(f"{backend}[{backend_time}]")
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

        # Select fastet backend for each operation
        logging.info("-------- Expected inference time ---------")
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
                    logging.debug("[{}] {} -> {} : {:.2f} ms decrease".format(
                        op_index, default_backend, op_backend,
                        (value[default_backend] - value[op_backend]) / 1000))

        for backend in backend_list:
            logging.info(f"{backend} backend : {backend_infer_time[backend]/1000:.2f} ms")
        logging.info(f"Backend scheduling : {schedule_time / 1000:.2f} ms")

        logging.info("-------- Backend Scheduling --------")
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
        logging.info(' '.join(cmd))

        # Create nnpackage with backend mapping
        dst_dir = Path(__file__).parent / 'nnpkg_sched' / self.nnpkg_dir.name
        self.nnpkg_helper.copy(self.nnpkg_dir, dst_dir)
        self.nnpkg_helper.add_config(dst_dir, cmd)
