# onnxkit

### Purpose

_onnxkit_ allows users to encode/decode ONNX model files.

### How to use

Currently it supports two operations, _decode_ and _encode_.

```
nncc$ path_to_onnxkit/onnxkit
ERROR: COMMAND is not provided

USAGE: path_to_onnxkit/onnxkit [COMMAND] ...

SUPPORTED COMMANDS:
  decode
  encode
```

`decode` reads a binary graphproto file and shows its textual form.

`encode` is the reverse of decode, it reads a textual graphproto file and prints
its binary form.

Each command can read from or print to the console or from/to a file if given
through the argument. First argument is used as an input file path and second
as a output file path. If second argument is omitted, output is the console.
To give the first argument as a console, please use `-`.

### Examples

Example to `decode`
```
nncc$ cat my_awesome_model.pb | path_to_onnxkit/onnxkit decode > decoded.pbtxt
```
```
nncc$ cat my_awesome_model.pb | path_to_onnxkit/onnxkit decode - decoded.pbtxt
```
```
nncc$ path_to_onnxkit/onnxkit decode my_awesome_model.pb > decoded.pbtxt
```
```
nncc$ path_to_onnxkit/onnxkit decode my_awesome_model.pb decoded.pbtxt
```

Above four examples for `decode` command gives the same result. This applies
to other commands.

Example to `encode`
```
nncc$ cat decoded.pbtxt | path_to_onnxkit/onnxkit encode > encoded.pb
```

### Dependency

- onnx
- Protobuf
- cli
