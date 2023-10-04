# q-implant

_q-implant_ writes quantization parameters and weights (given as .json and .npy files) to a circle model.

Therefore, before proceeding with q-implant, you should define `qparam.json` and `\*.npy` files so that they can be applied to the same operand as the circle you want to apply.

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
