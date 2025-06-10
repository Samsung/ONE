#!/usr/bin/python

# Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

import numpy as np
import sys
import json
import struct


def printUsage(progname):
    print("%s <.json>" % (progname))
    print("  This program extracts weight and bias values in TFLite format [N,H,W,C]")
    print("    to .npy files in ACL format [N,C,H,W]")
    print("  .npy filenames is set according to the layer's name")


if len(sys.argv) < 2:
    printUsage(sys.argv[0])
    exit()

filename = sys.argv[1]
f = open(filename)
j = json.loads(f.read())

tensors = j['subgraphs'][0]['tensors']
buffer_name_map = {}

for t in tensors:
    if 'buffer' in t:
        if t['buffer'] in buffer_name_map:
            print('find conflict!!')
            print(t)
            print(buffer_name_map)
        comps = t['name'].split('/')
        names = []
        if len(comps) > 1 and comps[0] == comps[1]:
            names = comps[2:]
        else:
            names = comps[1:]

        layername = '_'.join(names)

        shape = t['shape']
        buffer_name_map[t['buffer']] = {'name': layername, "shape": shape}

for i in range(len(j['buffers'])):
    b = j['buffers'][i]
    if 'data' in b:
        if i not in buffer_name_map:
            print(
                "buffer %d is not found in buffer_name_map. skip printing the buffer..." %
                i)
            continue

        filename = "%s.npy" % (buffer_name_map[i]['name'])
        shape = buffer_name_map[i]['shape']
        buf = struct.pack('%sB' % len(b['data']), *b['data'])

        elem_size = 1
        for s in shape:
            elem_size *= s

        l = struct.unpack('%sf' % elem_size, buf)
        n = np.array(l, dtype='f')
        n = n.reshape(shape)
        if len(shape) == 4:
            # [N,H,W,C] -> [N,C,H,W]
            n = np.rollaxis(n, 3, 1)
        elif len(shape) == 3:
            # [H,W,C] -> [C,H,W]
            n = np.rollaxis(n, 2, 0)
        elif len(shape) == 1:
            pass
        else:
            print("Undefined length: conversion skipped. shape=", shape)
        #print shape, filename, n.shape
        np.save(filename, n)

print("Done.")
