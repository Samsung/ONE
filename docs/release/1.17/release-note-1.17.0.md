# Release Note 1.17.0

## ONE Compiler

### Compiler Frontend

- More optimization pass
  - Remove Quant-Dequant sequence
  - Replace Sub with Add
  - Substitute StridedSlice to Reshape
  - Fuse Mean with Mean
  - Fuse Transpose with Mean
  - Substitute PadV2 to Pad
- Add new InstanceNorm pattern in `FuseInstanceNormPass`
- Add verbose option
- Introduce `onecc` driver to `one-cmds`
- Introduce `one-profile` driver to `one-cmds`

## ONE Runtime

### gpu_cl backend added

- New backend(gpu_cl) added. This backend exploits tensorflow lite's gpu delegate.
- This backends supports the following operations : Add, Convolution, Depthwise Convolution, Pooling, Reshape, Relu, Softmax 
