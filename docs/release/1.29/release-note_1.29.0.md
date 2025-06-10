# Release Note 1.29.0

## ONE Compiler

- Support more optimization option(s): `--transform_sqrt_div_to_rsqrt_mul`, `--fold_mul`,
    `--fuse_add_to_fullyconnected_bias`, `--fuse_mul_to_fullyconnected_weights`,
    `--fuse_mul_with_fullyconnected`.
- Add more optimization: `CanonicalizePass`.
- _tflite2circle_ supports more data types: FLOAT64, UINT64, UINT32, UINT16.
- _luci-interpreter_ supports more data types on some ops.
- Support multiple option names in command schema.
