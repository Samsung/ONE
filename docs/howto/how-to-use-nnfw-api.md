# How to Use NNFW API

## Prepare nnpackage

### Convert tensorflow pb file to nnpackage
Follow the [compiler guide](https://github.com/Samsung/ONE/blob/master/docs/nncc/v1.0.0/tutorial.md) to generate nnpackge from tensorflow pb file

### Convert tflite file to nnpackage
Please see [model2nnpkg](https://github.com/Samsung/ONE/tree/master/tools/nnpackage_tool/model2nnpkg) for converting from tflite model file.

## Build app with NNFW API

Here are basic steps to build app with [NNFW C API](https://github.com/Samsung/ONE/blob/master/runtime/onert/api/include/nnfw.h)

1) Initialize nnfw_session
``` c
nnfw_session *session = nullptr;
nnfw_create_session(&session);
```
2) Load nnpackage
``` c
nnfw_load_model_from_file(session, nnpackage_path);
```
3) (Optional) Assign a specific backend
``` c
  // Use 'acl_neon' backend only
  // Note that defalut backend is 'cpu'.
  nnfw_set_available_backends(session, "acl_neon");
```

4) Compilation
``` c
  // Compile model
  nnfw_prepare(session);
```

5) Prepare Input/Output
``` c
  // Prepare input. Here we just allocate dummy input arrays.
  std::vector<float> input;
  nnfw_tensorinfo ti;
  nnfw_input_tensorinfo(session, 0, &ti); // get first input's info
  uint32_t input_elements = num_elems(&ti);
  input.resize(input_elements);
  // TODO: Please add initialization for your input.
  nnfw_set_input(session, 0, ti.dtype, input.data(), sizeof(float) * input_elements);

  // Prepare output
  std::vector<float> output;
  nnfw_output_tensorinfo(session, 0, &ti); // get first output's info
  uint32_t output_elements = num_elems(&ti);
  output.resize(output_elements);
  nnfw_set_output(session, 0, ti.dtype, output.data(), sizeof(float) * output_elements);
```
6) Inference
``` c
  // Do inference
  nnfw_run(session);
```

## Run Inference with app on the target devices
reference app : [minimal app](https://github.com/Samsung/ONE/blob/master/runtime/onert/sample/minimal)

```
$ ./minimal path_to_nnpackage_directory
```
