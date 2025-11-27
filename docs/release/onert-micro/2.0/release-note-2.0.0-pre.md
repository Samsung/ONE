## Release Notes for onert-micro 2.0.0-pre

### Overall Structure Refactored

- c++ api has been changed : [onert-micro c++ api](https://github.com/samsung/ONE/blob/master/onert-micro/onert-micro/include/OMInterpreter.h)
- 60 ops supported : Abs, Add, AddN, AveragePool2D, ArgMax, ArgMin, Concatenation, BatchToSpaceD, Cos, Div, DepthwiseCov2D, Dequatize, FullyCoected, Cov2D, Logistic, Log, Gather, GatherD, Exp, Greater, GreaterEqual, ExpadDims, Equal, Floor, FloorDiv, FloorMod, Pad, Reshape, ReLU, ReLU6, Roud, Less, L2ormalize, L2Pool2D, LessEqual, LeakyReLU, LogSoftmax, Mul, Maximum, MaxPool2D, Miimum, otEqual, Si, SquaredDifferece, Slice, Sub, Split, SpaceToBatchD, StridedSlice, Square, Sqrt, SpaceToDepth, Tah, Traspose, TrasposeCov, Softmax, While, Rsqrt, Upack

### onert-micro supports on-device training feature

- Trainable Operations : 5 operations ( Conv2D, FullyConnected, MaxPool2D, Reshape, Softmax )
- Loss : MSE, Categorical Cross Entropy
- Optimizer : ADAM, SGD
- C api for training feature : [onert-micro c api header](https://github.com/samsung/ONE/blob/master/onert-micro/onert-micro/include/onert-micro.h)
- limitation : For now, you can train topologically linear model
