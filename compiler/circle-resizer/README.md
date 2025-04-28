# circle-resizer

_circle-resizer_ is a command-line tool designed to change the shapes of models inputs in `.circle` format.

The basic syntax of `circle-resizer` is:
```bash
./circle-resizer --input_path <input_model.circle> --output_path <output_model.circle> --input_shapes <shapes>
```

## Arguments:
- `--input_path`: Path to the input .circle model file (required).
- `--output_path`: Path to save the resized .circle model (required).
- `--input_shapes`: Comma-separated list of new input shapes in the format [dim1,dim2,...]. Example for two inputs: [1,2,3],[4,5] (required).
- `--version`: Display version information and exit.

## Example Command
```bash
./circle-resizer --input_path model.circle --output_path resized_model.circle --input_shapes [1,3,224,224],[10]
```
