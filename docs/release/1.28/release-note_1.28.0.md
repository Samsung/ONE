# Release Note 1.28.0

## ONE Runtime

### Python API

- Support experimental python API
  - Refer howto document for more details

### On-device Training

- Support on-device training with circle model
  - Training parameter can be passed to _onert_ via _onert_`s experimental API or loading new model format including training information: _circle_plus_
  - Trained model can be exported to _circle_ model via experimental API `nnfw_train_export_circle`
  - Supporting Transfer learning from a pre-trained circle model
- Introduce _circle_plus_gen_ tool
  - Generates a _circle_plus_ model file with a given training hyperparameters
  - Shows a _circle_plus model details

### Runtime configuration API
- _onert_ supports runtime configuration API for prepare and execution phase via experimental APIs
  - `nnfw_set_prepare_config` sets configuration for prepare phase, and `nnfw_reset_prepare_config` resets it to default value
  - `nnfw_set_execution_config` sets configuration for execution phase, and `nnfw_reset_execution_config` resets it to default value
  - Supporting prepare phase configuration: prepare execution time profile
  - Supporting execution phase configuration: dump minmax data, dump execution trace, dump execution time
- Introduce new API to set _onert_ workspace directory: `nnfw_set_workspace`
  - _onert_ workspace directory is used to store intermediate files during prepare and execution phase

### Minmax Recorder
- Now _onert_'s minmax recorder dumps raw file format instead of HDF5 format
- _onert_ dumps minmax data into workspace directory

### On-device Compilation
- _onert_ supports full quantization of uint8/int16 type weight and activation.
  - To quantize activation, _onert_ requires minmax data of activation.
- _onert_ supports on-device code generation for special backend requiring special binary format such as DSP, NPU.
  - Introduce new experimental API for code generation: `nnfw_codegen`

### Type-aware model I/O usage
- If loaded model is quantized model, _onert_ allows float type I/O buffer
  - _onert_ converts float type input buffer to quantized type internally
  - _onert_ fills float type output buffers by converting quantized type output data to float type internally
- On multimodel package, _onert_ allows edges between quantized model and float type model
