# locoex

_locoex_ is an extension of loco. Classes with the `COp` prefix enable *Custom Operation*.
In this version, a *custom operation* means one of the following:

1. an op that is supported by Tensorflow but not by moco and onert
2. an op that is not supported by Tensorflow, moco or onert

`COpCall` node will represent an IR entity that calls custom operations and kernels.
