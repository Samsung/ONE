#! /usr/bin/python
import argparse
import json
import numpy as np
import sys
import subprocess
import time
from pareto import ParetoData
from profile_args import ProfileArgs
from runner import Runner
from utils import progressbar

if __name__ == "__main__":
    t_start = time.time()
    parser = ProfileArgs("random_sampler.py", description="Random sampler")
    parser.add_argument(
        '--iterations', type=int, default=100, help='Number of iterations')

    # Parse arguments
    args = parser.parse_args()
    dumpfile = args.dumpfile
    iterations = args.iterations

    # Initialize a runner and Pareto data structure obj
    runner = Runner(args.model, args.run_folder, args.backends, args.mode)
    pareto_obj = ParetoData()
    # Initialize variables for random sampler
    n_assignments = runner.get_solution_spacelen()
    n_iterations = min(iterations, n_assignments)
    chk_ptr = 0
    marked_samples = {}

    # Profile at random over solution space
    for r in range(n_iterations):
        random_sample = int(np.random.rand() * n_assignments)
        while random_sample in marked_samples:
            random_sample = int(np.random.rand() * n_assignments)
        marked_samples[random_sample] = True
        if args.mode == "name":
            exec_time, max_rss = runner.profile_by_opname(random_sample)
        elif args.mode == "index":
            exec_time, max_rss = runner.profile_by_opindex(random_sample)
        else:
            print("Invalid mode ", mode)
            sys.exit(-1)

        pareto_obj.update_pareto_solutions(random_sample, exec_time, max_rss)
        progressbar(r, n_assignments, prefix="% samples computed. : ")
    progressbar(r + 1, n_assignments, prefix="% samples computed. : ")

    # Dump profiler results
    dumpdata = {}
    dumpdata['mode'] = args.mode
    dumpdata = pareto_obj.dump_pareto_solutions(dumpdata)
    dumpdata = runner.dump_config(dumpdata)
    with open(dumpfile, "w") as ofile:
        json.dump(dumpdata, ofile)
    t_end = time.time()
    print("\n")
    print("done.., profiling time = ", (t_end - t_start), " seconds")
