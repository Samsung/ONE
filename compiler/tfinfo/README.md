# tfinfo

This dir contains a helper classes to handle `test.info` files under `res/TensorFlowTests`.

## Format of 'test.info' file

Each line should contain the following fields:
- `input` or `output`
- node_name:digits
- type (see enum TF_DataType in tensorflow/c/c_api.h)
- [ shapes ]
   - In case of scalar, use '[ ]' as shape
