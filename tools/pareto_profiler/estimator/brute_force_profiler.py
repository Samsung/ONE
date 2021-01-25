#! /usr/bin/python
import argparse
import json
import sys
from profile_args import ProfileArgs
from runner import Runner
from utils import progressbar

if __name__ == "__main__":
    parser = ProfileArgs(
        prog="brute_force_profiler.py", description="Profiles nnpackage_run using oplist")
    # Parse arguments
    args = parser.parse_args()
    modelfile = args.model
    mode = args.mode
    n_backends = args.backends
    dumpfile = args.dumpfile

    # Initialize a runner for given model and target
    runner = Runner(args.model, args.run_folder, args.backends, args.mode)
    nruns = runner.get_solution_spacelen()
    profile_results = {}
    profile_results['solutions'] = []
    chk_ptr = 0

    # Profile each backend setting, record execution time and peak memory
    for r in range(nruns):
        if (r % 100) == 0:
            # Checkpointing results, in case the runs take too long
            if chk_ptr > 0:
                with open("/tmp/solutions.json") as ifile:
                    tmp_results = json.load(ifile)

                with open("/tmp/solutions.json", "w") as ofile:
                    json.dump(tmp_results + profile_results['solutions'][chk_ptr:], ofile)
            else:
                with open("/tmp/solutions.json", "w") as ofile:
                    json.dump(profile_results['solutions'], ofile)
            chk_ptr = r

        if args.mode == "name":
            exec_time, max_rss = runner.profile_by_opname(r)
        elif args.mode == "index":
            exec_time, max_rss = runner.profile_by_opindex(r)
        else:
            print("Invalid mode ", mode)
            sys.exit(-1)

        profile_results['solutions'].append({
            "time": exec_time,
            "memory": max_rss,
            "id": r
        })
        progressbar(r, nruns, prefix="% samples computed. : ")
    progressbar(nruns, nruns, prefix="% samples computed. : ")

    oplist, opmap, opname_by_indx = runner.get_opconfig()

    if args.mode == "index":
        profile_results['oplist'] = oplist
        profile_results['opmap'] = opmap
        profile_results['opname_by_indx'] = opname_by_indx
    elif args.mode == "name":
        profile_results['oplist'] = oplist
    else:
        print("Invalid mode ", mode)
        sys.exit(-1)

    with open(dumpfile, "w") as ofile:
        json.dump(profile_results, ofile)
    print "\nDone.."
