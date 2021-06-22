#! /usr/bin/python

import argparse
import os
import Graph

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "test_partition.py", description="Example code to partition models")
    parser.add_argument("modelfile", type=str, help="TFLite file with path")
    parser.add_argument("tracefile", type=str, help="Chrome trace file with path")
    parser.add_argument("--num_parts", type=int, default=2, help="Number of partitions")
    parser.add_argument(
        "--num_runs", type=int, default=10, help="Number of runs (topological orderings)")

    # Parse arguments
    args = parser.parse_args()

    # Partition
    g = Graph.GraphTopology(args.modelfile, args.tracefile)
    g.partition_minmax_multiple(K=args.num_parts, nruns=args.num_runs)
