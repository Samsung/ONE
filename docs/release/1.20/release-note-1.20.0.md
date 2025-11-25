# Release Note 1.20.0

## ONE Compiler

### Compiler Frontend

- luci-interpreter supports multiple kernels with PAL layer including Cortext-M
- luci-interpreter supports integer tensor for partly kernels
- luci import support constant without coping to reduce memory for luci-interpreter
- Reduce duplicate codes to package released modules
- Limited support for ONNX LSTM/RNN unrolling while importing
- Limited support for ARM32 cross build
- Support new operator: SVDF
- New virtual CircleVariable to support tensor with variable
- Support quantization of BatchMatMul Op
- Support mixed(UINT8 + INT16) quantization
- Support backward propagation of quantization parameters
- Upgrade default python to version 3.8
- Support TensorFlow 2.8.0, ONNX-TF 1.10.0, ONNX 1.11.0
- Upgrade circle schema to follow tflite schema v3b
- Refactor to mio-tflite280, mio-circle04 with version and helpers methods
- Use one flatbuffers 2.0 version
- Drop support for TensorFlow 1.x
- Fix for several bugs, performance enhancements, and typos

## ONE Runtime

### Introduce TRIX backend
- TRIX backend supports trix binary with NHWC layout
- TRIX backend supports trix binary with input/output of Q8 and Q16 type

### API supports new data type
- Symmetric Quantized int16 type named "NNFW_TYPE_TENSOR_QUANT16_SYMM_SIGNED"
