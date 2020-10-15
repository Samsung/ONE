# onnx-tools

_onnx-tools_ provides developer tools to support ONNX format in compiler frontend.

## onnx-dump.py

Use `onnx-dump.py` to dump ONNX model graph in human readable text format.

For example,

```
[General] -----------------------------
IR version = 6
Producer   = pytorch 1.6

[Operators] ---------------------------
    3 Conv
    3 Relu
...

[Initializers] ------------------------
"0.bias"        FLOAT [16]
"0.weight"      FLOAT [16, 1, 3, 3]
...

[Nodes] -------------------------------
Conv("Conv_0")
    A dilations: [1, 1], group: 1, kernel_shape: [3, 3], pads: [1, 1, 1, 1], strides: [2, 2]
    I "input.1"
    I "0.weight"
    I "0.bias"
    O "7"
Relu("Relu_1")
    I "7"
    O "8"
...

[Graph Input/Output]-------------------
    I: "input.1"       FLOAT [1, 1, 28, 28]
    O: "21"            FLOAT [1, 10]
```

In `[Nodes]` section, `A` is for attributes for the node, `I` for input name and `O` for output name.

`I` and `O` also applies to `[Graph Input/Output]` section.

## onnx-ops.py

Use `onnx-ops.py` to dump ONNX model operators.

You can use with other command line tools to analyze operators in the model file.

For example,
```bash
$ python onnx-ops.py mymodel.onnx | sort | uniq -c
      1 Concat
      1 Constant
      3 Conv
      1 Gather
      1 GlobalAveragePool
      3 Relu
      1 Reshape
      1 Shape
      1 Unsqueeze
```
