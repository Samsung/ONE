# TFLite model file tools

TFLite model file analyzers.

Highly recommand to use `uv` to run these tools.

## Prepare environment

You need generated python code from TFLite schema by flatc. `pre-sync.sh` is provided to do this.

`pre-sync.sh` builds flatc and generate python code from TFLite schema.

You can build flatc by yourself or download pre-built binary. If you have pre-built binary, you can skip building flatc step by setting `FLATC_BIN` variable in `pre-sync.sh`. But we hightly recommand to use same version of flatc described in `pyproject.toml`'s flatbuffers dependency.

After code generation, you can prepare python environment by `uv sync` command.

```
./pre-sync.sh
uv sync
```

## Model parser

### Purpose

This tool print operators, tensors, and buffers information in tflite model file (`.tflite`)

### How to use

```
uv run model-parser <model-file>
```

### Example

```
$ uv run model-parser /home/nnfw/convolution_test.tflite

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
uv run select-operator <base model file> <opcode list txt file> <output file name>
```

### Example

```
$ cat /home/nnfw/opcodelist.txt
107 108 109 110 111 112 113 114 115 116 117 118 119 120

$ uv run select-operator /home/nnfw/inceptionv3_non_slim_2015.tflite \
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

$ uv run select-operator multi_subgraph.tflite
opcodelist.txt test.tflite -g 1
```

Above selects operator index 11, 12, 13 in subgraph 1

### Generating separate models for multi-model from a model file by using model generator here

To make one model multi-model, separate models and inputs/outputs information of each model are required.
So run model generator with the option `--store-io-info`

#### How to use

```
uv run select-operator <base model file> <opcode list txt file> <output file name> --store-io-info <output json file name>
```

#### Example

This example generates one model into two separate models.

```
$ cat 0-26.txt
0-26

$ cat 27-30.txt
27-30

$ uv run select-operator mobilenet_v1_1.0_224.tflite 0-26.txt m1.tflite --store-io-info m1.json
Input tensor(s): [81]
Output tensor(s): [44]
Append subgraphs, orginal index :  0 , new index :  0

$ uv run select-operator mobilenet_v1_1.0_224.tflite 27-30.txt m2.tflite --store-io-info m2.json
Input tensor(s): [6]
Output tensor(s): [7]
Append subgraphs, orginal index :  0 , new index :  0

$ cat m1.json
{"org-model-io": {"inputs": {"new-indices": [81]}, "outputs": {"new-indices": [-1]}}, "new-model-io": {"inputs": {"org-indices": [88], "new-indices": [81]}, "outputs": {"org-indices": [50], "new-indices": [44]}}}

$ cat m2.json
{"org-model-io": {"inputs": {"new-indices": [-1]}, "outputs": {"new-indices": [7]}}, "new-model-io": {"inputs": {"org-indices": [50], "new-indices": [6]}, "outputs": {"org-indices": [87], "new-indices": [7]}}}

```
The meaning of `m1.json` above is as follows:
- original model has 1 input and 1 output
  - The only input is located at tensors[81] from new model.
  - The only output has new-index -1, which means it is not in new-model.
- new-model has 1 input and 1 output
  - The only input was located at tensors[88] from org model, and it is located at tensors[81] from new model.
  - The only output was located at tensors[50] from org model, and it is located at tensors[44] from new model.

With the model files and inputs/outputs infomation files generated above, you can use `model2nnpkg.py` to create nnpkg for multi-model.

## Colaboration model parser and model generator

1. Get imformation about base model using model parser
2. Select operators you want to make test model
3. Make text file including selected operators index
4. Generate test model file using model generator
