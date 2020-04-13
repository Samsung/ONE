#!/usr/bin/python3

import os
import sys

try:
    from caffe2.python import workspace, model_helper
    from caffe2.python.predictor import mobile_exporter
except ImportError:
    print("!! Caffe2 not installed, caffe2 frontend test not generated", file=sys.stderr)
    exit(1)


def save_net(init_net_pb, predict_net_pb, model):
    init_net, predict_net = mobile_exporter.Export(workspace, model.net, m.params)
    with open(predict_net_pb, 'wb') as f:
        f.write(model.net._net.SerializeToString())
    with open(init_net_pb, 'wb') as f:
        f.write(init_net.SerializeToString())


resDir = sys.argv[1]

m = model_helper.ModelHelper(name='unsupported_net')
m.net.GivenTensorFill([], 'input_data', values=(1., ), shape=(1, ))
m.net.Sin(['input_data'], 'result')
save_net(os.path.join(resDir, 'init_net.pb'), os.path.join(resDir, 'predict_net.pb'), m)
