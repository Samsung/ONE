#! /usr/bin/python

# Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

import argparse
import utils
import sys
import json
import time
from Hlps import Hlps
from profile_args import ProfileArgs
from runner import Runner


def hlps_profiler(modelfile,
                  run_folder,
                  num_backends=2,
                  mode="name",
                  nruns=3,
                  num_samples=2000,
                  dumpfile=None):
    runner = Runner(modelfile, run_folder, num_backends, mode=mode)
    hlps = Hlps(runner, num_backends=num_backends, num_samples=num_samples)

    config_set = set()
    sample_cnt = 0
    total_reject_list = []

    for r in range(nruns):
        config_set, sample_cnt_iter = hlps.hlps_routine(config_set)
        sample_cnt += sample_cnt_iter

    # Add the index mode search here.
    print("Starting search over extended space")
    print("\n")
    if mode == "index":
        hlps.enable_extended_search()
        for r in range(nruns):
            config_set, sample_cnt_iter = hlps.hlps_routine(config_set)
            sample_cnt += sample_cnt_iter

    # Export results to json file
    # Dump profiler results
    dumpdata = {}
    dumpdata['mode'] = args.mode
    dumpdata['sample_cnt'] = sample_cnt
    dumpdata = hlps.dump_results(dumpdata)
    with open(dumpfile, "w") as ofile:
        json.dump(dumpdata, ofile)


if __name__ == "__main__":
    t_start = time.time()
    parser = ProfileArgs(
        "hlps_on_device.py",
        description="On-Device Optimizing Profiler for TensorFlowLite Models")
    parser.add_argument(
        '--iterations',
        type=int,
        default=3,
        help='Number of iterations, less than 10 should be enough')
    parser.add_argument(
        '--samples', type=int, default=2000, help='Number of samples per iteration')
    parser.add_argument(
        '--offline',
        type=bool,
        default=False,
        help='Set to True for running over profiled data')
    parser.add_argument('--profiled_data', type=str, help='Profile file with path')

    args = parser.parse_args()

    hlps_profiler(
        args.model,
        args.run_folder,
        num_backends=args.backends,
        mode=args.mode,
        nruns=args.iterations,
        num_samples=args.samples,
        dumpfile=args.dumpfile)
    t_end = time.time()
    with open(args.dumpfile, "r") as ifile:
        dumpdata = json.load(ifile)
    dumpdata['profiling time'] = (t_end - t_start)
    with open(args.dumpfile, "w") as ofile:
        json.dump(dumpdata, ofile)
    print("done.., profiling time = ", (t_end - t_start), " seconds")
