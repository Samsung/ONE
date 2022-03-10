# circle-inspect

_circle-inspect_ allows users to retrieve various information from a Circle model file

## Information to inspect

Operators with `--operators`
- show operator codes one line at a time in execution order

Example
```
$ circle-inspect --operators model.circle
```

Result
```
RESHAPE
DEPTHWISE_CONV_2D
ADD
```

To get the count of specific operator, use other tools like sort, uniq, etc.

Operators with `--tensor_dtype`
- show name and dtype of each tensor one line at a time

Example
```
$ circle-inspect --tensor_dtype quantized_conv2d.circle
```

Result
```
ifm UINT8
weights UINT8
bias INT32
ofm UINT8
```
