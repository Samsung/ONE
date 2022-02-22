#! /usr/bin/python
import json
from queue import LifoQueue


def get_runtime_per_operation(trace_file):
    with open(trace_file) as ifile:
        data = json.load(ifile)
    traceEvents = data['traceEvents']
    time_val = {}
    stack = LifoQueue(maxsize=1000)
    for t in traceEvents:
        if t == {}:
            continue
        if (t["name"].lower() != "graph" and "permute" not in t["name"].lower()) and \
          ("subg" not in t["name"].lower() and "permute" not in t["name"].lower()):
            if t["ph"] == "B":
                stack.put((t["name"], int(t["ts"])))
            elif t["ph"] == "E":
                opname, st_time = stack.get()
                assert (opname == t["name"])
                if "$" in t["name"]:
                    time_val[int(
                        t["name"].split(" ")[0].lstrip('$'))] = int(t["ts"]) - st_time
                else:
                    time_val[int(
                        t["name"].split(" ")[0].lstrip('@'))] = int(t["ts"]) - st_time

    time_idx = [y for x, y in (sorted(time_val.items(), key=lambda item: item[0]))]
    return time_idx
