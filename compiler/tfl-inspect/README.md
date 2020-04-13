# tfl-inspect

_tfl-inspect_ allows users to retrieve various information from a TensorFlow Lite model files

## Information to inspect

#### --operators

Operators with `--operators`
- show operator codes one line at a time in execution order

Example
```
$ tfl_inspect --operators model.tflite
```

Result
```
RESHAPE
DEPTHWISE_CONV_2D
ADD
```

To get the count of specific operator, use other tools like sort, uniq, etc.

Example
```
$ tfl-inspect --operators inception_v3.tflite | sort | uniq -c
```
Result
```
     10 AVERAGE_POOL_2D
     15 CONCATENATION
     95 CONV_2D
      4 MAX_POOL_2D
      1 RESHAPE
      1 SOFTMAX
```

#### --conv2d_weight

Conv2D series weight input node type with `--conv2d_weight`
- shows Conv2D series node weight input node type
- Conv2D series: CONV2D, DEPTHWISE_CONV_2D

Example result
```
CONV2D,CONST
DEPTHWISE_CONV_2D,RELU
CONV2D,CONST
```
