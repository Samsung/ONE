#!/usr/bin/python3

import argparse, logging, sys
from backend_profiler import BackendProfiler
from backend_scheduler import BackendScheduler


def main(args):
    if args.profile:
        backend_profiler = BackendProfiler(args.user, args.ip, args.nnpackage,
                                           args.num_threads)
        backend_profiler.sync()
        backend_profiler.profile()
    backend_scheduler = BackendScheduler(args.nnpackage, args.num_threads)
    backend_scheduler.schedule()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(add_help=False)
    required = arg_parser.add_argument_group('required arguments')
    optional = arg_parser.add_argument_group('optional arguments')

    # Add back help
    optional.add_argument('-h',
                          '--help',
                          action='help',
                          default=argparse.SUPPRESS,
                          help='show this help message and exit')
    required.add_argument("--nnpackage",
                          type=str,
                          required=True,
                          help="nnpackage folder to profile")
    required.add_argument("--ip",
                          type=str,
                          required=True,
                          help="IP address of remote client")
    optional.add_argument("-n",
                          "--num_threads",
                          type=int,
                          default=1,
                          help="Number of threads used by one runtime")
    optional.add_argument("-u", "--user", type=str, help="User of remote client")
    optional.add_argument("-v",
                          "--verbose",
                          action='store_const',
                          dest="verbose_level",
                          default=logging.INFO,
                          const=logging.DEBUG,
                          help="Print verbose message")
    optional.add_argument("--no-profile",
                          dest='profile',
                          action='store_false',
                          help="Disable profiling")
    optional.set_defaults(profile=True)
    args = arg_parser.parse_args()

    logging.basicConfig(stream=sys.stdout,
                        level=args.verbose_level,
                        format="[%(levelname).5s] %(message)s")

    main(args)
