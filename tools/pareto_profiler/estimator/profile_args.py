#! /usr/bin/python
import argparse


class ProfileArgs(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(ProfileArgs, self).__init__(args, kwargs)
        self.add_argument(
            'model', type=str, default=None, help='nnpackage name with path')
        self.add_argument('run_folder', type=str, help="path to nnpackage_run executable")
        self.add_argument(
            '--mode',
            type=str.lower,
            choices=["index", "name"],
            default="name",
            help='Profile by operation index or name')
        self.add_argument('--backends', type=int, default=2, help='Number of backends')
        self.add_argument(
            '--dumpfile',
            type=str.lower,
            default="/tmp/final_result.json",
            help='JSON Dumpfile name with path')
