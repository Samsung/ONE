# tflite2nnpkgtc

`tflite2nnpkgtc` is a tool to convert tflite to nnpackage test case.
It takes `tflite` as input and generates `nnpackage` + golden data for value test.

## prerequisite

Install tensorflow >= 1.12. It is tested with tensorflow 1.13, 1.14 and 2.0.

## usage

```
$ ./tflite2nnpkgtc.sh -h
Usage: tflite2nnpkgtc.sh [options] tflite
Convert tflite to nnpkg-tc

Returns
     0       success
  non-zero   failure

Options:
    -h   show this help
    -o   set output directory (default=.)

```

## example
```
# @ host
$ tools/nnpackage_tool/sth2nnpkgtc/tflite2nnpkgtc.sh -o nnpkg-tcs cast.tflite
Generating nnpackage cast in nnpkg-tcs

# then, nnpkg is generated in $outdir/$basename
$ tree nnpkg-tcs/cast
nnpkg-tcs/cast
├── cast.tflite
└── metadata
    ├── MANIFEST
    └── tc
        ├── expected.h5
        └── input.h5

# @ target
# run nnpkg with nnpackage_run and compare with h5diff
$ tests/scripts/nnpkg_test.sh -i nnpkg-tcs cast
```
