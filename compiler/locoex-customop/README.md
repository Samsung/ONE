# locoex

_locoex_ is an extention of loco. Classes with `COp` prefix enables *Custom Operation*.
In this version, a *custom operation* means one of the following:

1. an op that is supported by Tensorflow but not supported both by the moco and the onert
1. an op that is not supported by Tensorflow, moco, and the onert

`COpCall` node will represent IR entity that calls custom operations and kernels.
