#!/usr/bin/python3
"""
Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import caffe
import numpy as np
import sys
import h5py
from itertools import chain
from caffe import layers as L
import random
import lmdb
from collections import Counter, OrderedDict

if (len(sys.argv) < 2):
    dest_folder = ''
    print('Using current directory as destination folder')
else:
    dest_folder = sys.argv[1] + '/'


class PH:
    """
    PlaceHolder value
    """

    def __init__(self, type, param):
        self.type = type
        self.param = param


# Bookkeeping
LS = 224
# bynaryProto file for Infogain
H = np.eye(3, dtype='f4')
blob = caffe.io.array_to_blobproto(H.reshape((1, 1, 3, 3)))
with open(dest_folder + 'infogainH.binaryproto', 'wb+') as f:
    f.write(blob.SerializeToString())

# List of hdf5 files
with open(dest_folder + "in", 'w+') as f:
    f.write('in.hdf5')

#Window File
with open(dest_folder + "in_winds", 'w+') as f:
    f.write("""# 1
in.jpg
3
224
224
2
1 0.1 50 50 60 70
1 0.9 30 30 50 50
# 2
in.jpg
3
224
224
2
1 0.1 50 50 70 70
1 0.9 30 30 50 50
""")

# HDF5 file for HDF5DataSet
h5f = h5py.File(dest_folder + "in.hdf5", "w")
h5f.create_dataset("data", data=np.random.rand(1, 3, LS, LS))
h5f.close()

# LMDB file
env = lmdb.open(dest_folder + 'test-lmdb')
with env.begin(write=True) as txn:
    img_data = np.random.rand(3, LS, LS)
    datum = caffe.io.array_to_datum(img_data, label=1)
    txn.put('{:0>10d}'.format(1).encode('ascii'), datum.SerializeToString())
env.close()

# recurring parameters
losspara = {'ignore_label': True, 'normalization': 1, 'normalize': True}
softmaxpara = {'engine': 0, 'axis': 1}
gdfil = {'type': 'gaussian', 'std': 0.001}
cofil = {'type': 'constant', 'value': 0}
rp = {
    'num_output': 1,
    'weight_filler': gdfil,
    'bias_filler': cofil,
    'expose_hidden': True
}

filler_par = {
    'type': 'constant',
    'value': 0,
    'min': 0,
    'max': 1,
    'mean': 0,
    'std': 1,
    'sparse': -1,  # -1 means no sparsification
    'variance_norm': 0
}  # 0 = FAN_IN, 1 = FAN_OUT, 2 = AVERAGE

OPS = [
    ('Parameter', {
        'shape': {
            'dim': [1]
        },
        "is_data": True
    }),  # ok
    (
        'Data',
        {
            'source': 'test-lmdb',  # FIXME: unknown DB backend
            'batch_size': 1,
            'rand_skip': 0,
            'backend': 1,  # 0 = LEVELDB, 1 = LMDB
            'scale': 1.0,  # deprecated in favor of TransformationParameter
            'mean_file': 'wtf.is_that',
            'crop_size': 0,
            'mirror': False,
            'force_encoded_color': False,
            'prefetch': 4,
            "is_data": True
        }),
    (
        'DummyData',
        {
            'data_filler': cofil,  # ok
            #'num' : [1,1,1], # deprecated shape specification
            #'channels' : [2,2,2],
            #'height' : [3,3,3],
            #'width' : [4,4,4]},
            'shape': {
                'dim': [1, 3, LS, LS]
            },
            "is_data": True
        }),
    (
        'ImageData',
        {
            'source': 'in_imgs',  # file with list of imgs
            'top': 'op2',
            'batch_size': 1,
            'rand_skip': 0,
            'shuffle': False,
            'new_height': 0,
            'new_width': 0,
            'is_color': True,
            'root_folder': '',
            'scale': 1.0,  # deprecated in favor of TransformationParameter
            'mirror': False,
            "is_data": True
        }),
    (
        'WindowData',
        {
            'source': 'in_winds',
            'top': 'op2',
            'batch_size': 1,
            'mean_file': 'in.jpg',
            'transform_param': {
                'scale': 0.8,
                'crop_size': 24,
                'mirror': False,
                #'fg_treshold' : 0.5,
                #'bg_treshold' : 0.5,
                #'fg_fraction' : 0.25,
            },
            'context_pad': 1,
            'crop_mode': 'warp',
            'cache_images': True,
            'root_folder': './',
            "is_data": True
        }),
    (
        'HDF5Data',
        {
            'source': 'in',  # This is the name of the file WITH HDF5 FILENAMES 0_0
            # Top should have the same name as the dataset in the hdf5 file
            # FIXME Requires Caffegen to be built with Caffe that supports LMDB
            'batch_size': 1,
            'shuffle': False,
            "is_data": True
        }),
    ('Input', {
        'shape': {
            'dim': [1, 2, 3, 4]
        },
        "is_data": True
    }),  # ok
    (
        'MemoryData',
        {
            'batch_size': 1,  # ok
            'channels': 2,
            'height': 3,
            'width': 4,
            'top': "foo",
            "is_data": True
        }),

    ## Regular OPS
    (
        "Convolution",
        {
            'num_output': 64,  # ok
            'kernel_size': 9,
            'stride': 1,
            'pad': 0,
            'weight_filler': gdfil,
            'param': [{
                'lr_mult': 1
            }, {
                'lr_mult': 0.1
            }],
            'bias_filler': cofil
        }),

    # Depthvise conv
    (
        "Convolution",
        {
            'num_output': 12,  # ok
            'kernel_size': 9,
            'stride': 1,
            'dilation': 2,
            'group': 3,
            'pad': 0,
            'weight_filler': gdfil,
            'param': [{
                'lr_mult': 1
            }, {
                'lr_mult': 0.1
            }],
            'bias_filler': cofil
        }),
    (
        "Deconvolution",
        {
            'convolution_param':  # ok
            {
                'num_output': 4,
                'kernel_size': 9,
                'stride': 1,
                'pad': 0,
                'weight_filler': gdfil,
                'bias_filler': cofil
            }
        }),
    # Depthvise deconv
    (
        "Deconvolution",
        {
            'convolution_param':  # ok
            {
                'num_output': 12,
                'kernel_size': 9,
                'stride': 1,
                'dilation': 2,
                'group': 3,
                'pad': 0,
                'weight_filler': gdfil,
                'bias_filler': cofil
            }
        }),
    (
        'BatchNorm',
        {
            'eps': 1e-5,  # ok
            'moving_average_fraction': 0.999
        }),
    (
        'LRN',
        {
            'alpha': 1.,  # ok
            'beta': 0.75,
            'norm_region': 1,
            'local_size': 5,
            'k': 1,
            'engine': 0
        }),
    # local_size[default 5]: the number of channels to sum over
    # alpha[default 1]: the scaling paramete
    # beta[default5]: the exponent
    # norm_region[default ACROSS_CHANNLS]: whether to sum over adjacent channels(ACROSS_CHANNLS) or nearby
    # spatial locations(WITHIN_CHANNLS)
    # `input / (1 + (\alpha/n) \sum_i x_i^2)^\beta`
    (
        "MVN",
        {
            'normalize_variance': True,  # ok
            'across_channels': False,
            'eps': 1e-9
        }),
    (
        'Im2col',
        {
            'convolution_param':  # ok
            {
                'num_output': 64,
                'kernel_size': 9,
                'stride': 1,
                'pad': 0,
                'weight_filler': gdfil,
                # 'param' : [{'lr_mult':1},{'lr_mult':0.1}],
                'bias_filler': cofil
            }
        }),
    ('Dropout', {
        'dropout_ratio': 0.5
    }),  # ok
    ('Split', {}),  # ok
    ('Concat', {
        'axis': 1
    }),  # ok
    (
        'Tile',
        {
            'axis': 1,  # ok
            'tiles': 2
        }),
    ('Slice', {
        'axis': 1,
        'top': 'op2',
        'slice_point': 1
    }),
    (
        'Reshape',
        {
            'shape': {
                'dim': [1, 0, -1]
            },  # ok
            'axis': 0,
            'num_axes': -1
        }),
    # reshapes only [axis, axis + num_axes] if those aren't 0 and -1; axis can be negative
    # 0 in shape means retaining dim size, -1 means auto size
    (
        'Flatten',
        {
            'axis': 1,  # ok
            'end_axis': -1
        }),
    (
        'Pooling',
        {
            'pool': 0,  # ok # pool: 0 = MAX, 1 = AVE, 2 = STOCHASTIC
            'pad': 0,  # can be replaced with pad_w, pad_h
            'kernel_size': 3,  # can be replaced with kernel_w, kernel_h
            'stride': 1,  # can be replaced with stride_w, stride_h
            'engine': 0,
            'global_pooling': False
        }),
    # 'round_mode' : 0}), # 0 = CELS, 1 = FLOOR
    (
        'Reduction',
        {
            'operation': 1,  # ok # 1 = SUM, 2 = ASUM, 3 = SUMSQ, 4 = MEAN # ok
            'axis': 0,
            'coeff': 1.0
        }),
    (
        'SPP',
        {
            'pyramid_height': 1,  # ok
            'pool': 0,
            'engine': 0
        }),
    (
        'InnerProduct',
        {
            'num_output': 2,  # ok
            'bias_term': True,
            'weight_filler': filler_par,
            'bias_filler': filler_par,
            'axis': 1,
            'transpose': False
        }),
    (
        'Embed',
        {
            'num_output': 2,  # ok
            'input_dim': 1,
            'bias_term': True,
            'weight_filler': filler_par,
            'bias_filler': filler_par
        }),
    (
        'ArgMax',
        {
            'out_max_val': False,  # ok # if True, outputs pairs (argmax, maxval) # ok
            'top_k': 1,
            'axis': -1
        }),
    (
        'Softmax',
        {
            'engine': 0,  # ok
            'axis': 1
        }),
    (
        'ReLU',
        {
            'negative_slope': 0,  # ok
            'engine': 0
        }),
    (
        'PReLU',
        {
            'filler': filler_par,  # ok
            'channel_shared': False
        }),
    ('ELU', {
        'alpha': 1
    }),  # ok
    ('Sigmoid', {
        'engine': 0
    }),  # ok
    ('BNLL', {}),  # ok
    ('TanH', {
        'engine': 0
    }),  # ok
    ('Threshold', {
        'threshold': 0
    }),  # ok
    (
        'Bias',
        {
            'axis': 0,  # ok
            'num_axes': -1,
            'filler': filler_par
        }),
    (
        'Scale',
        {
            'axis': 0,  # ok
            'num_axes': -1,
            'filler': filler_par,
            'bias_term': False,
            'bias_filler': filler_par
        }),
    ('AbsVal', {}),  # ok
    (
        'Log',
        {
            'base': -1.0,  # ok
            'scale': 1.0,
            'shift': PH(float, (2.0, 10.0)),
            'how_many': 10
        }),  # y = ln(shift + scale * x) (log_base() for base > 0)
    (
        'Power',
        {
            'power': -1.0,  # ok
            'scale': 1.0,
            'shift': 0.0
        }),  # y = (shift + scale * x) ^ power
    (
        'Exp',
        {
            'base': -1.0,  # ok
            'scale': 1.0,
            'shift': 0.0
        }),

    ## TWO INPUTS
    (
        'Crop',
        {
            'axis': 2,  # ok
            'offset': [0],
            "inputs": 2
        }),  # if one offset - for all dims, more - specifies
    (
        "Eltwise",
        {
            'operation': 1,  # ok
            'coeff': [3, 3],
            'stable_prod_grad': True,
            "inputs": 2
        }),
    ("EuclideanLoss", {
        "inputs": 2
    }),  # ok
    ("HingeLoss", {
        'norm': 1,
        "inputs": 2
    }),  # L1 = 1; L2 = 2; # ok
    ("SigmoidCrossEntropyLoss", {
        'loss_param': losspara,
        "inputs": 2
    }),  # ok

    ## TWO Inputs, special shape
    (
        "Accuracy",
        {
            'top_k': 1,  # FIXME: different bottom shapes needed
            'axis': 0,
            'ignore_label': 0,
            "inputs": 2,
            "special_shape": [1, 3, 1, 1]
        }),
    (
        "SoftmaxWithLoss",
        {
            'loss_param': losspara,  # FIXME: different bottom shapes needed
            'softmax_param': softmaxpara,
            "inputs": 2,
            "special_shape": [1, 1, 1, 1]
        }),
    ("MultinomialLogisticLoss", {
        'loss_param': losspara,
        "inputs": 2,
        "special_shape": [1, 1, 1, 1]
    }),  # FIXME: different bottom shapes needed
    ("Filter", {
        "inputs": 2,
        "special_shape": [1, 1, 1, 1]
    }),  # FIXME: different bottom shapes needed
    ('BatchReindex', {
        "inputs": 2,
        "special_shape": [2]
    }),  # takes indices as second blob
    ("InfogainLoss", {
        'source': 'infogainH.binaryproto',
        'axis': 1,
        "inputs": 2,
        "special_shape": [1, 1, 1, 1]
    }),
    (
        'Python',
        {
            'python_param':  # Custom Loss layer
            {
                'module': 'Pyloss',  # the module name -- usually the filename -- that needs to be in $PYTHONPATH
                'layer': 'EuclideanLossLayer',  # the layer name -- the class name in the module
                'share_in_parallel': False
            },
            # set loss weight so Caffe knows this is a loss layer.
            # since PythonLayer inherits directly from Layer, this isn't automatically
            # known to Caffe
            'loss_weight': 1,
            "inputs": 2,
            "special_shape": [1, 3, 1, 1]
        },
    ),

    ## NOTOP OPS
    ('HDF5Output', {
        'file_name': 'out.hdf5',
        "inputs": 2,
        "is_notop": True
    }),  # ok
    ('Silence', {
        "inputs": 2,
        "is_notop": True
    }),  # ok, need to remove tops

    ## THREE INPUTS
    ("RNN", {
        'recurrent_param': rp,
        'top': "out2",
        "inputs": 3
    }),  # ok
    ("Recurrent", {
        'recurrent_param': rp,
        'top': "out2",
        "inputs": 3
    }),  # ok

    ## FOUR INPUTS
    ("LSTM", {
        'recurrent_param': rp,
        'top': ["out2", "out3"],
        "inputs": 4
    }),  # ok

    ## Handled explicitly (special case)
    ("ContrastiveLoss", {
        'margin': 1.0,
        'legacy_version': False
    }),
]

#Helper functions


def traverse(obj, callback=None):
    """
     walks a nested dict/list recursively
    :param obj:
    :param callback:
    :return:
    """
    if isinstance(obj, dict):
        value = {k: traverse(v, callback) for k, v in obj.items()}
    elif isinstance(obj, list):
        value = [traverse(elem, callback) for elem in obj]
    else:
        value = obj

    if callback is None:
        return value
    else:
        return callback(value)


def mock(inp):
    if not (isinstance(inp, PH)): return inp
    if inp.type == int:
        return random.randint(*inp.param)
    if inp.type == float:
        return random.uniform(*inp.param)


EXTRA_SHAPES = \
    [(), # alredy defined
     [1, 3],
     [1, 3, 1],
     [1, 3, 1]]


class Layer:
    """
    Represents a caffe layer
    """

    def __init__(self, name, params):
        self.name = name
        self.args = params
        if self.args == None: self.args = dict()
        self.num_inp = self.args.pop("inputs", 1)
        self.num_out = self.args.pop("outputs", 1)
        self.special_shape = self.args.pop("special_shape",
                                           False)  # 2nd input has special shape
        self.is_data = self.args.pop("is_data", False)
        self.is_notop = self.args.pop("is_notop", False)

    def make_net(self):
        """
        Creates a protobuf network
        :return:
        """
        net = caffe.NetSpec()

        if self.is_data:
            net.data = getattr(L, self.name)(**self.args)

        # Very special,
        elif self.name == "ContrastiveLoss":
            net.data = L.Input(shape={'dim': [1, 4]})
            net.data1 = L.DummyData(data_filler=cofil, shape={'dim': [1, 4]})
            net.data2 = L.DummyData(data_filler=cofil, shape={'dim': [1, 1]})

            net.op = getattr(L, self.name)(net.data, net.data1, net.data2, **self.args)

        # this covers most cases
        else:
            net.data = L.Input(shape={'dim': [1, 3, LS, LS]})
            if self.num_inp == 2:
                net.data1 = L.DummyData(data_filler=cofil, shape={'dim': [1, 3, LS, LS]})
            elif self.num_inp > 2:
                for i in range(1, self.num_inp):
                    setattr(
                        net, "data" + str(i),
                        L.DummyData(data_filler=cofil, shape={'dim': EXTRA_SHAPES[i]}))
            if self.special_shape:
                net.data = L.Input(shape={'dim': [1, 3, 1, 1]})
                net.data1 = L.DummyData(
                    data_filler=cofil, shape={'dim': self.special_shape})

            net.op = getattr(L, self.name)(
                net.data,
                *[getattr(net, "data" + str(i))
                  for i in range(1, self.num_inp)], **self.args)

        if self.is_notop:
            net.op.fn.tops = OrderedDict()
            net.op.fn.ntop = 0  # the messing about in question

        return net


class LayerMaker:
    """
    Factory class for Layer
    """

    def __init__(self, params):
        self.name, self.args = params
        self.how_many = self.args.pop("how_many", 1)

    def make(self):
        return [Layer(self.name, traverse(self.args, mock)) for i in range(self.how_many)]


layer_gen = chain(*map(lambda para: LayerMaker(para).make(), OPS))

filename = dest_folder + '{}_{}.prototxt'

counter = Counter()
for layer in layer_gen:
    n = layer.make_net()
    counter[layer.name] += 1

    with open(filename.format(layer.name, counter[layer.name] - 1), 'w+') as ptxt_file:
        print(n.to_proto(), file=ptxt_file)

    if layer.name == "Python":  # Special case for python layer
        with open("Python_0.caffemodel", 'wb+') as caffemodelFile:
            caffemodelFile.write(n.to_proto().SerializeToString())
