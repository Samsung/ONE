# q-implant

_q-implant_ writes quantization parameters and weights (given as .json and .npy files) to a circle model.

## Format of input files (`.json` and `.npy`)

The main input file is a `.json` file, which is a dictionary.

The dictionary's key is a tensor name, and its value is quantization parameters and values(if exists).

The `.npy` file is a NumPy file that is generated through the `numpy.save` function. 

When filling the value of `quantization parameters` or `values`(acting like a key inside `<tensor_name>`) in the param.json, the `.npy` file must be saved.

```
{
  <tensor_name>: {
    "dtype": <dtype>,
    "scale": <path/to/scale_npy>,
    "zerop": <path/to/zerop_npy>,
    "quantized_dimension": <dim>,
    "value": <path/to/value_npy>
  },
  ...
}
```
`<tensor_name>`: String (target tensor name)

`<dtype>`: String (data type of the target tensor. ex: "uint8" or "int16")

`<path/to/scale_npy>`: String (path to the .npy file that contains scale. The npy file has to be 1d array of fp32.)

`<path/to/zerop_npy>`: String (path to the .npy file that contains zerop. The npy file has to be 1d array of int64.)

`<dim>`: Integer (quantized dimension)

`<path/to/value_npy>`: String (path to the .npy file that contains zerop. The npy file should have the same shape/dtype with the target tensor.)

NOTE "value" is an optional attribute. It is only necessary for weights.

NOTE `.npy` files have to be placed in the same directory with `qparam.json`.

## Example

```sh
q-implant input.circle qparam.json output.circle
```

`qparam.json` and `*.npy` files must exist in the same directory.

For example, there are four operands in Conv2D_000.circle: `ifm`, `ker`, `bias`, and `ofm`.

In this regard, `qparam.json` and `*.npy` can be defined as follows.(`*.npy` internal values are random values for testing)

- qparam.json

```json
{
  "ifm": {
    "dtype": "uint8",
    "scale": "0.npy",
    "zerop": "1.npy",
    "quantized_dimension": 0
  },
  "ker": {
    "dtype": "uint8",
    "scale": "2.npy",
    "zerop": "3.npy",
    "quantized_dimension": 0,
    "value": "4.npy"
  },
  "bias": {
    "dtype": "int32",
    "scale": "5.npy",
    "zerop": "6.npy",
    "quantized_dimension": 0,
    "value": "7.npy"
  },
  "ofm": {
    "dtype": "uint8",
    "scale": "8.npy",
    "zerop": "9.npy",
    "quantized_dimension": 0
  }
}
```

- \*.npy

```
0.npy : [0.27102426]
1.npy : [216]
2.npy : [0.44855267]
3.npy : [103]
4.npy : [[[[237 157]]]]
5.npy : [0.03877867]
6.npy : [97]
7.npy : [88]
8.npy : [0.83054835]
9.npy : [7]
```
