# Release Note 1.9.0

## ONE Compiler

### Compiler supports more operations

- NonMaxSuppressionV4, NonMaxSuppressionV5, PadV2, Unique

### Changes

- Quantization enhancements: channel wise UINT8 quantization(Conv2D, DepwiseConv, TransposeConv, FullyConnected)
- Experimental requantization from INT8 to UINT8
- Adding more operator value tests
- tf2tfliteV2 supports conversion from Keras model, saved model
- Refactoring for better maintenance long Class codes using visitor patterns 
- Introducing optimization pass that fuses batch normalization with Transposed Convolution.


## ONE Runtime

### Runtime backend operation support

- CPU backend: RANK
- CPU backend qasymm uint8: LOG_SOFTMAX
- ACL-CL backend: LEAKY_RELU, RESIZE_NEAREST_NEIGHBOR


### Optimization

- Copy Elimination between compatible backends

### Operation Implementation

- Operations with same parameters are unified

### Change

- CPU backend qasymm uint8 performance enhancement: arithmetic operations
