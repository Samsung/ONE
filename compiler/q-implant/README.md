# q-implant

_q-implant_ writes quantization parameters and weights (given as .json and .npy files) to a circle model.

## Example

```sh
q-implant input.circle qparam.json output.circle
```

`qparam.json` and `*.npy` files must exist in the same directory. For example,

```json
{
  "ifm": {
    "dtype": "uint8",
    "scale": "0.npy",
    "zerop": "1.npy",
    "quantized_dimension": 0
  },
  ...
}
```

`0.npy` and `1.npy` must exist in the same directory with `qparam.json`.
