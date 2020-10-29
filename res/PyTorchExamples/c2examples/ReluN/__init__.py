import numpy as np

import onnx
from caffe2.python import core, workspace, model_helper
from caffe2.proto import caffe2_pb2

workspace.GlobalInit(
    [
        "caffe2",
        "--caffe2_log_level=0",
    ]
)

# model
# _model_ = core.Net("OpTestNet")
# op = core.CreateOperator("Sin", ["X"], ["Y"])
# _model_.Proto().op.extend([op])

# X = np.random.rand(2, 2).astype(np.float32)
# workspace.FeedBlob("X", X)
helper = model_helper.ModelHelper(name="TestNet")
op_relun = helper.net.ReluN(["X"], "Y", n=0.3)

# print(str(helper.net.Proto()))
# workspace.RunNetOnce(helper.param_init_net)
# workspace.CreateNet(helper.net)
# workspace.RunNet(helper.name)
# Y = workspace.FetchBlob("Y")

_model_ = helper.net
_model_init_ = core.Net("InitNet")

# value info for onnx generation
data_type = onnx.TensorProto.FLOAT
data_shape = (2, 2)
_value_info_ = { 'X': (data_type, data_shape) }
