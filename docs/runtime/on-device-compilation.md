# On-Device Compilation

ONERT supports on-device compilation - on-device quantization and on-device code generation.

## On-device quantization

ONERT supports on-device quantization for the **float32** model. On-device quantization has two mode - full quantization and weight-only quantization.

### Weight-only quantization

Weight-only quantization quantizes only weights of the model. The activation is still in float32 precision. This mode is useful when the model size reduction is more important than the inference speedup.

For weight-only quantization, follow below steps:
- Load float32 model
- Set quantization type for weight-only quantization
- Set path to save quantized model
- Call quantize API to perform quantization

```cpp
// Load float32 model
nnfw_load_model_from_file(session, pkg_path);

// Set quantization type: weight-only, symmetric, int8
nnfw_set_quantization_type(session, NNFW_QUANTIZE_TYPE_WO_I8_SYM);

// Set path to save quantized model
nnfw_set_quantized_model_path(session, quantized_model_path);

// Quantize model
nnfw_quantize(session);

// Run model for inference with quantized model
nnfw_run(session);
```

When the model is quantized, you can use the quantized model because the quantized model is loaded automatically after quantization. You don't need to load the quantized model explicitly.


### Full quantization

Full quantization quantizes both weights and activations of the model. This mode is useful when specific runtime backend requires quantized model. To quantize activation, runtime should gather information about activation range during the execution of the model. Therefore, it needs to run the model enough times to get accurate activation range.

For full quantization, follow below steps:

- Load float32 model
- Gather activation range by running the model multiple times
  - Prepare model to run
  - Set input and output buffer(s)
  - Set execution configuration to gather activation range
  - Run model multiple times for inference with gathering activation range
- Quantize model if activation range is gathered enough
  - Set quantization type for full quantization
  - Set path to save quantized model
  - Call quantize API to perform quantization

```cpp
// Load float32 model
nnfw_load_model_from_file(session, pkg_path);

// Prepare model to run
nnfw_prepare(session);

// Set input and output buffer(s)
nnfw_set_input(session, input_index, input_type, input_buffer, input_element_size);
nnfw_set_output(session, output_index, output_type, output_buffer, output_element_size);

// Set execution configuration to gather activation range
nnfw_set_execute_config(session, NNFW_RUN_CONFIG_DUMP_MINMAX, nullptr);

// Run model multiple times for inference with gathering activation range
for (int i = 0; i < num_of_inference; ++i)
{
  nnfw_run(session);
}

// Set quantization type: full, asymmetric, uint8
nnfw_set_quantization_type(session, NNFW_QUANTIZE_TYPE_U8_ASYM);

// Set path to save quantized model
nnfw_set_quantized_model_path(session, quantized_model_path);

// Quantize model
nnfw_quantize(session);

// Reset execution configuration to normal execution
nnfw_reset_execute_config(session);

// Run model for inference with quantized model
nnfw_run(session);
```

When the model is quantized, you can use the quantized model because the quantized model is loaded automatically after quantization. You don't need to load the quantized model explicitly. Also, you don't need to set input and output buffers for the quantized data type because runtime automatically casts input and output buffers data between float32 and quantized data type. But you can set input and output buffers for the quantized data type after model full quantization if you want to use them directly without data casting.

## On-device code generation

ONE supports on-device code generation. On-device code generation generates backend-specific code from the model and saves it as a supported file format. This feature is useful when the backend requires a specific precompiled model file format.

### Prerequisites

To use on-device code generation, you need to install plugin that supports on-device code generation. On-device code generation plugin must fulfill interface defined in `ICodegen.h`.

Plugin should be installed in `{libdir}/nnfw/codegen` with `lib<filetype>-gen.so` name pattern. For example, if your plugin generates file with `.abc` extension, then plugin library should be named `libabc-gen.so`.

### Usage

To generate code, follow below steps:

- Load model
- (Optional) Set path to save generated code
  - If path is not set, generated code will be saved in same directory with model with same name but target name extension
- Call generate_code API to perform code generation

```cpp
// Load model
nnfw_load_model_from_file(session, pkg_path);

// (Optional) Set path to save generated code
// nnfw_set_codegen_model_path(session, codegen_model_path);

// Generate code for target backend: target codegen plugin name is "abc" (installed as `libabc-gen.lib`)
nnfw_codegen(session, "abc-gen", NNFW_CODEGEN_PREF_DEFAULT);

// Prepare model to run
nnfw_prepare(session);

// Set backend to use generated code on specific target backend if need
nnfw_set_available_backend(session, "abc");

// Set input and output buffer(s)
nnfw_set_input(session, input_index, input_type, input_buffer, input_element_size);
nnfw_set_output(session, output_index, output_type, output_buffer, output_element_size);

// Run model
nnfw_run(session);
```

## Collaboration on-device quantization and code generation

On-device quantization and code generation can be used together when target backend requires quantized model and specific precompiled model file format.

## Test tool support

On-device compilation is supported in test tools `onert_run`

### Quantization

Example: weight-only quantization
- Input file: `test.circle`
- Quantization type: weight-only, symmetric, int8
- Output file: `test.q.circle`

```sh
$ onert_run --quantize int8_wo \
    --qpath test.q.circle \
    test.circle
```

Example: full quantization
- Input file: `test.circle`
- Quantization type: full, asymmetric, uint8
- Output file: `test.q.circle`
- Number of inference to gather activation range: 10

```sh
$ onert_run -- quantize uint8 \
    --qpath test.q.circle \
    --minmax_run 10 \
    test.circle
```

### Code generation

Example
- Input file: `test.circle`
- Target backend: `abc_back`
- Target plugin name: `abc`
- Output file: `test.abc`

```sh
$ BACKENDS='abc_back' onert_run --codegen abc-gen \
    --cpath test.abc \
    test.circle
```

### Quantization and code generation

Example
- Input file: `test.circle`
- Quantization type: full, asymmetric, uint8
- Number of inference to gather activation range: 10
- Quantized model file: `test.q.circle`
- Target backend: `abc_back`
- Target plugin name: `abc`
- Codegen output file: `test.abc`

```sh
$ BACKENDS='abc_back' onert_run --quantize uint8 \
    --qpath test.q.circle \
    --minmax_run 10 \
    --codegen abc-gen \
    --cpath test.abc \
    test.circle
