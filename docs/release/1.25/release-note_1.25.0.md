# Release Note 1.25.0

## ONE Runtime

- Support ubuntu 20.04

### CPU Backend Operation
- CPU backend supports per-channel hybrid quantization of int8 type weight and float activation. (TFLite's dynamic range quantization)

### On-device Quantization
- _onert_ supports new experimental API for on-device quantization.
- As a 1st step, _onert_ supports per-channel hybrid quantization of int8/int16 type weight and float activation.
- API requires file path to export quantized model.

### Minmax Recorder
- _onert_` support minmax recording of each layer as experimental feature. It is not supported by API yet.
- Output file format is HDF5. (File format may change later).
