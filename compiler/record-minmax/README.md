# record-minmax

_record-minmax_ is a tool to embed min/max values of activations to the circle model for post-training quantization.

## Usage

This will run with the path to the input model (.circle), input data (.h5), and the output model (.circle).

```
$ ./record-minmax <path_to_input_model> <path_to_input_data> <path_to_output_model>
```

For example,
```
$ ./record-minmax input.circle input.h5 out.circle
```

Output is a circle model where min/max values of activation tensors are saved in QuantizationParameters.
