# pb2nnpkgtc

`pb2nnpkgtc` is a tool to select a subgraph of pb model and convert to `nnpackage` with golden data and generated pb.

It takes `pb` as input and generates `nnpackage`.

## prerequisite

Install tensorflow >= 1.12. It is tested with tensorflow 1.13, 1.14 and 2.0.

Install node. (Any version will do. I recommend you to use `nvm`.)

Set environmet variables from usage below.

## usage

```
$ ./pb2nnpkgtc.sh -h
Usage: pb2nnpkgtc.sh [options] pb inputs outputs
Convert pb to nnpkg-tc

Returns
     0       success
  non-zero   failure

Options:
    -h   show this help
    -o   set output directory (default=.)

Environment variables:
   flatc           path to flatc
                   (default=./build/externals/FLATBUFFERS/build/flatc)
   tflite_schema   path to tflite schema (i.e. schema.fbs)
   circle_schema   path to tflite schema (i.e. schema.fbs)
```

## example
```
# @ host
$ tools/nnpackage_tool/sth2nnpkgtc/pb2nnpkgtc.sh test_model.pb img_placeholder conv2d_transpose

# then, nnpkg is generated in {basename}.{outputname}
# it contains all of pb, tflite, circle, and golden data.

$ tree test_model.conv2d_transpose
test_model.conv2d_transpose
├── test_model.conv2d_transpose.circle
├── test_model.conv2d_transpose.pb
├── test_model.conv2d_transpose.tflite
└── metadata
    ├── MANIFEST
        └── tc
                ├── expected.h5
                        └── input.h5

# @ target
$ OP_BACKEND_ALLOPS=cpu \
onert/test/onert-test nnpkg-test test_model.conv2d_transpose
[  Run  ] ./test_model.out   Pass
[Compare] ./test_model.out   Pass
```
