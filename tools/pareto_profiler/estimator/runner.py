#! /usr/bin/python

import json
import numpy as np
from utils import fetch_config_by_name
from utils import fetch_config_by_indx
from utils import generate_vars
from utils import generate_vars_for_indx
from utils import exec_shell
from utils import import_configs
import sys


class Mapper:
    def __init__(self, opmap, oplist, opname_by_index):
        self._opmap = opmap
        self._oplist = oplist
        self._opname_by_indx = opname_by_index

    def get_oplist(self):
        return self._oplist

    def get_opmap(self):
        return self._opmap

    def get_opname_by_indx(self):
        return self._opname_by_indx


class Runner:
    def __init__(self, model, run_folder, num_backends, mode):
        self._model = model
        self._run_folder = run_folder
        self._mode = mode
        oplist, opmap, opname_by_index = import_configs(mode)
        self._mapper = Mapper(opmap, oplist, opname_by_index)
        self._nbackends = num_backends

    def get_solution_spacelen(self):
        if self._mode == "name":
            return self._nbackends**len(self._mapper.get_oplist())
        elif self._mode == "index":
            return self._nbackends**max(self._mapper.get_opmap())
        else:
            print("Unknown mode ", mode, ", exiting profiler")
            sys.exit(-1)

    def run_inference(self, solution):
        cmd_str = [
            ". /tmp/envvars.sh && " + self._run_folder + "/nnpackage_run -w1 -r1 -m1 -l "
            + self._model + "/metadata/tc/input.h5 " + self._model + " 2> /dev/null"
        ]
        res = exec_shell(cmd_str, newline_split=True)
        try:
            exec_time = float(res[4].split(' ')[-2])
            max_rss = int(res[13].split(' ')[-2])
        except IndexError:
            print("got index error at config ", solution)
            print("result: ", res)
            print("####")
            sys.exit(-1)
        return (exec_time, max_rss)

    def profile_by_opname(self, solution):
        generate_vars(self._mapper.get_oplist(), solution, self._nbackends)
        return self.run_inference(solution)

    def profile_by_opindex(self, solution):
        generate_vars_for_indx(self._mapper.get_opmap(), solution, self._nbackends)
        return self.run_inference(solution)

    def get_opconfig(self):
        return self._mapper.get_oplist(), self._mapper.get_opmap(
        ), self._mapper.get_opname_by_indx()

    def dump_config(self, dumpdata):
        if self._mode == "name":
            dumpdata.update({'oplist': self._mapper.get_oplist()})
        elif self._mode == "index":
            dumpdata.update({'oplist': self._mapper.get_opmap()})

        configs = {}
        for solution in dumpdata['solutions']:
            if self._mode == "name":
                configs[int(solution["id"])] = fetch_config_by_name(
                    dumpdata['oplist'], solution["id"], self._nbackends)
            elif self._mode == "index":
                configs[int(solution["id"])] = fetch_config_by_indx(
                    dumpdata['oplist'], solution["id"], self._nbackends)
        dumpdata.update({'configs': configs})
        return dumpdata
