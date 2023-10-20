## minmax-embedder-value-test

minmax-embedder-value-test aims to test minmax-embedder tool.

It generates minmax data (encoded min and max from run idx, op/input index).

Then, it checks whether it is correctly embedded into circle.

minmax-embedder is supposed to be executed in a device.

Thus, test is also implemented so that it can be run on a device (especially
on Tizen device. For example, It does not use Python.

### minmax-data-gen

`minmax-data-gen` generates minmax-data for test.

#### Usage

```
Usage: ./minmax-data-gen [-h] [--num_inputs NUM_INPUTS] [--num_ops NUM_OPS] minmax

[Positional argument]
minmax    	path to generated minmax data

[Optional argument]
-h, --help  	Show help message and exit
--num_inputs	number of input layers (default:1)
--num_ops   	number of operators (default:1)
```
