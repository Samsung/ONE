# Release Note 1.30.0

## ONE Compiler

- Support more optimization option(s): `--dynamic_batch_to_single_batch`, `--fuse_rmsnorm`, 
 `--fuse_rope`, `--fuse_rsqrt`.
- Introduce _one-import-pytorch_ for direct PyTorch import that converts PyTorch modules 
 straight to Circle.
- Support MX (microscaling) data type.
- Introduce _circle-mlir_ that converts new ops (e.g., ConvTranspose2D, DepthwiseConv2D, 
 ArgMax, Reduce*, Resize, Pooling, GELU, Sigmoid, Softmax, Slice/StridedSlice, Select, 
 Gather, BatchMatMul) and validates shape inference.
- ONNX is converted from new _circle-mlir/onnx2circle_ tool (onnx-tf tool is deprecated)
- _luci-interpreter_ handles scalar indices in Gather and additional element types across 
 several kernels.
- Ubuntu22.04 with python3.10 is official supported platform, python3.10 is necessary 
 for Ubuntu20.04 and Ubuntu24.04 with python3.12 is experimentally supported.
- Python packages are upgraded to TensorFlow 2.19.0, ONNX 1.18.0, ONNXRuntime 1.21.1, 
 Torch 2.7.0.

## ONE Runtime

### General Updates

- Ubuntu version support changes:
  - Add 24.04 LTS support
  - Deprecate 20.04 LTS support

### Runtime API Updates

- NNAPI is not official API anymore. It is still supported but used for test purpose only.
- Model loading from model file without NN package is supported by C/Python APIs. Supporting 
 model types are TFLite, Circle, and TVN.
- Python API supports training APIs.
- Python API supports dynamic shapes.
- Supporting custom BroadcastTo, AddV2, BatchMatMul are deprecated. These custom operations 
 loading was introduced to support custom operations by compiler but now it is replaced with
 general operations.

### Runtime Core Updates

- Layout conversion is supported for input and output only. It does not support for 
 intermediate tensors.
- HDF5 dependency on minmax dumper is removed. Now it uses runtime's own tensor dumping 
 functions.
- Block quantization type (GGML_Q4_0, GGML_Q8_0) is supported for quantized models.
- RMSNorm, RoPE and GELU operations are supported.
- Einsum and MatrixBandPart operations are deprecated. They were introduced to support
 custom operation by compiler but now they are replaced with general operations.

### Backend Kernel Updates

- CPU backend supports INT64 BinaryArithmetic operations.
- CPU backend supports FullyConnected block quantized weight operations.
- CPU backend supports Gather block quantized constant input operations.
- CPU backend supports RMSNorm, RoPE, and GELU operations.
- ARM Compute Library version is updated to 24.07.

### On-device Training Updates

- Memory usage is enhanced during training by usedef analysis and tensor planning.
- Checkpoint is introduced to export and load tensor and optimizer data, making it 
 easier to save and restore training states.

### On-device Compilation Updates

- Auto-compilation is supported when compilation threshold is met.
