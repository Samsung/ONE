# circle-interpreter

_circle-interpreter_ is a end user tool to infer a Circle model file.

## Information with arguments

Three positional arguments are required.

```bash
# Usage:
circle-interpreter <circle_model> <input_prefix> <output_prefix>
```

- `circle_model`: Path to the Circle model to infer
- `input_prefix`: Path to input data file. n-th input data will be read from a file named `${input_prefix}n` (e.g. `add.circle.input0`)
- `output_prefix`: Path to output data file. n-th output data will be write to a file named `${output_prefix}n` (e.g. `add.circle.output0`)
