## Model parser

### Purpose

This tool print operators, tensors, and buffers information in tflite model file (`.tflite`)

### How to use

```
./model_parser.py <model file>
```

### Example

```
$ ./tools/tflitefile_tool/model_parser.py /home/nnfw/convolution_test.tflite

[Main model]

Main model input tensors: [0]
Main model output tensors: [1]
Operators list

Operator 0: CONV_2D
        Input Tensors[0 3 2]
        Output Tensors[1]


Tensor-Buffer mapping & shape

Tensor    0 : buffer    3 |  Empty | FLOAT32 | Shape [1, 299, 299, 3] (Mul)
Tensor    1 : buffer    4 |  Empty | FLOAT32 | Shape [1, 149, 149, 32] (conv)
Tensor    2 : buffer    1 | Filled | FLOAT32 | Shape [32] (conv/Conv2D_bias)
Tensor    3 : buffer    2 | Filled | FLOAT32 | Shape [32, 3, 3, 3] (conv/conv2d_params)

$
```

## Model generator from other model file

### Purpose

This tool makes small model file from base model file (such as inception v3)

### How to use

```
./select_operator.py <base model file> <opcode list txt file> <output file name>
```

### Example

```
$ cat /home/nnfw/opcodelist.txt
107 108 109 110 111 112 113 114 115 116 117 118 119 120

$ ./tools/tflitefile_tool/select_operator.py /home/nnfw/inceptionv3_non_slim_2015.tflite \
/home/nnfw/opcodelist.txt /home/nnfw/test.tflite

Input tensor(s): [29]
Output tensor(s): [31]

$ Product/out/bin/tflite_run /home/nfs/inception_test.tflite
nnapi error: unable to open library libneuralnetworks.so
input tensor indices = [29,]
Input image size is smaller than the size required by the model. Input will not be set.
output tensor indices = [31(max:567),]
Prepare takes 0.000516954 seconds
Invoke takes 0.719677 seconds

$
```

You can use range such as `107-120` in `opcodelist.txt` instead of using each operator index

### Subgraph

You can select subgraph to select operator. Default subgraph index is 0.
If selected operators contain controlflow operator, the model to be generated will contain subgraphs of the selected controlflow operator.

```
$ cat /home/nnfw/opcodelist.txt
11-13

$ ./tools/tflitefile_tool/select_operator.py multi_subgraph.tflite
opcodelist.txt test.tflite -g 1
```

Above selects operator index 11, 12, 13 in subgraph 1

## Colaboration model parser and model generator

1. Get imformation about base model using model parser
2. Select operators you want to make test model
3. Make text file including selected operators index
4. Generate test model file using model generator
