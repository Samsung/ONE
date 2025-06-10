## Release Notes for onert-micro 2.0.0

### Operations updated

- New ops supported : Cast, Ceil, Elu, Fill
- New CMSIS NN accelrated ops : SVDF, Relu, Relu6

### New features for on-device training

- New Trainable Operation : GRU, StridedSlice
  - limitation : You can train GRU's weights. Since input gradient is not supported now, GRU layer should be the last layer for training.
- New Loss Function : Sparse Categorical Cross Entropy(experimental)
