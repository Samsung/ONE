#! /usr/bin/python
import argparse
import tensorflow as tf
import sys
sys.path.append("../estimator")
import subprocess
import os
import json
from functools import reduce
from utils import exec_shell
"""
  Generates from a tflite model, a list of unique onert operation names used in the model
"""


def generate_oplist_by_name(tflite_file):
    with open("operations_map.json") as ifile:
        data = json.load(ifile)
    op_dict = data['op_dict']

    intr = tf.lite.Interpreter(tflite_file)
    intr.allocate_tensors()
    tf_opset = set(op['op_name'] for op in intr._get_ops_details())
    try:
        onert_ops = set([op_dict[op] for op in tf_opset])
    except KeyError:
        print("Invalid mapping, check your tensorflow ops for new/unknown mappings: ",
              tf_opset)
        sys.exit(-1)
    return onert_ops


"""
  Returns the total data size for the model graph node (inputs + outputs)
  Params:
  op: operation instance (obtained from _get_ops_details())
  tsr: tensor instance (obtained from get_tensor_details()) 
"""


def get_op_data_size(op, tsr):
    data_size = 0
    for idx in op['inputs']:
        if tsr[idx]['shape'].size > 0:
            data_size += reduce(lambda x, y: x * y,
                                tsr[idx]['shape']) * tsr[idx]['shape'].dtype.itemsize

    for idx in op['outputs']:
        if tsr[idx]['shape'].size > 0:
            data_size += reduce(lambda x, y: x * y,
                                tsr[idx]['shape']) * tsr[idx]['shape'].dtype.itemsize
    return data_size


"""
  Generates from a tflite model, the following outputs:
  1.  opmap - a symbol/bit index mapping from every graph operation to a unique <operation name, data size> index identifier. This mapping
      will be used later when profiling the model at runtime.

  2.  oplist - a list of unique onert operation names used in the model

  3.  opname_by_index - a list of onert operation names, indexed by their topological order in the model
"""


def generate_oplist_by_name_size(tflite_file):
    intr = tf.lite.Interpreter(tflite_file)
    intr.allocate_tensors()
    ops = intr._get_ops_details()
    tsr = intr.get_tensor_details()

    opset = set()
    oplist = set()
    indx = []
    opname_by_indx = []
    # Fetch tensorflow operation mapping to onert kernels
    with open("operations_map.json") as ifile:
        data = json.load(ifile)
    op_dict = data['op_dict']

    # Fetch all unique operation names and <operation name, tensordata size> pairs
    for op in ops:
        opset.add((op['op_name'], get_op_data_size(op, tsr)))
        oplist.add(op_dict[op['op_name']])
        indx.append(op['index'])
    opname_by_indx = [op_dict[ops[i]['op_name']] for i in indx]

    # Create a 'm' bit/symbol map indexed by <opname, tensordata size> values
    inv_opset_map = {}
    i = 0
    for op in opset:
        inv_opset_map[op] = i
        i += 1

    # Map 'n' operation symbol space to 'm' <opname, tensordata size> space
    op_map = []
    for op in ops:
        data_size = get_op_data_size(op, tsr)
        op_map.append(inv_opset_map[(op['op_name'], data_size)])

    return op_map, oplist, opname_by_indx


"""
Script to generate oplist, given the following details:
1. Modelfile
2. target device type
3. Additional information, such as authetication for file tranfer

Info: python gen_oplist.py --help
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='''gen_backend: Generates oplist and uploads to target''',
        epilog="""Success.""")
    parser.add_argument(
        '--auth', type=str, default=None, help='authentication: <user@host>')
    parser.add_argument(
        '--mode',
        type=str.lower,
        choices=["index", "name"],
        default="name",
        help='Profile by operation index or name')
    parser.add_argument('model', type=str, default=None, help='tflite name with path')
    parser.add_argument(
        'target',
        type=str.lower,
        choices=['tizen', 'odroid'],
        default="odroid",
        help='target name')

    # Parse arguments
    args = parser.parse_args()
    modelfile = args.model
    target = args.target
    mode = args.mode
    if target == "odroid":
        auth_str = args.auth
        if auth_str is None:
            print("Need valid authentication")
            sys.exit(-1)

    # Generate oplist
    if mode == "name":
        opset = generate_oplist_by_name(modelfile)
        print(opset)
        with open('/tmp/oplist.json', 'w') as opfile:
            data = {}
            data['oplist'] = list(opset)
            json.dump(data, opfile)
    elif mode == "index":
        data = {}
        opmap, oplist, opname_by_indx = generate_oplist_by_name_size(modelfile)
        data['opmap'] = opmap
        data['oplist'] = list(oplist)
        data['opname_by_indx'] = opname_by_indx
        with open('/tmp/oplist.json', 'w') as opfile:
            json.dump(data, opfile)
    # Upload oplist to target
    if target == "tizen":
        exec_shell("sdb push /tmp/oplist.json /tmp/oplist.json")
    elif target == "odroid":
        print("auth_str = ", auth_str)
        exec_shell("scp  /tmp/oplist.json " + auth_str + ":/tmp/oplist.json")
    print("done...")
