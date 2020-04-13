# tfkit

## What is tfkit?

`tfkit` is a tool for manipulating TensorFlow model files.

## Tutorial: How to use?

Currently it supports two operations, _decode_ and _encode_.

```
nncc$ path_to_tfkit/tfkit
ERROR: COMMAND is not provided

USAGE: path_to_tfkit/tfkit [COMMAND] ...

SUPPORTED COMMANDS:
  decode
  encode
  unpack
  pack
```

`decode` reads a binary graphdef file and shows its textual form.

`encode` is the reverse of decode, it reads a textual graphdef file and prints
its binary form.

`unpack` decodes tensor value in byte encoded string in `tensor_content` field
to human readable list of float values. currently only supports textual
graphdef files.

`pack` is the reverse of unpack. this can be used to change the values for
debugging. also currently only supports textual graphdef files.

Each command can read from or print to the console or from/to a file if given
through the argument. First argument is used as an input file path and second
as a output file path. If second argument is omitted, output is the console.
To give the first argument as a console, please use `-`.

### Examples

Example to `decode`
```
nncc$ cat my_awesome_model.pb | path_to_tfkit/tfkit decode > decoded.pbtxt
```
```
nncc$ cat my_awesome_model.pb | path_to_tfkit/tfkit decode - decoded.pbtxt
```
```
nncc$ path_to_tfkit/tfkit decode my_awesome_model.pb > decoded.pbtxt
```
```
nncc$ path_to_tfkit/tfkit decode my_awesome_model.pb decoded.pbtxt
```

Above four examples for `decode` command gives the same result. This applies
to other commands.

Example to `encode`
```
nncc$ cat decoded.pbtxt | path_to_tfkit/tfkit encode > encoded.pb
```

Example to `unpack`
```
nncc$ cat packed.pbtxt | path_to_tfkit/tfkit unpack > unpacked.pbtxt
```

Example to `pack`
```
nncc$ cat unpacked.pbtxt | path_to_tfkit/tfkit pack > packed.pbtxt
```
