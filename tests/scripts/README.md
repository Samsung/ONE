# How the test driver works

## Unittest
- There are two kinds of unittest:
    - Kernel ACL
    - Runtime
- Related file : `unittest.sh`
- Usage :
```
$ ./tests/scripts/test-driver.sh \
    --artifactpath=. \
    --unittest
```
- The `unittest.sh` usage :

```
$ ./tests/scripts/unittest.sh \
    --reportdir=report \
    --unittestdir=Product/out/unittest
```

### Kernel ACL Unittest
- Test whether the various operations are performed successfully and whether the output and the expected value are the same.
- TC location : `libs/kernel/acl/src/`

### Runtime Unittest
- Test whether the expected value and the actual output value are the same when the model is configured, compiled and executed.
- TC location : `runtimes/tests/neural_networks_test/`

## Framework test
- Execute the **tflite model** using the given **driver**.
- There is a TC directory for each model, and a `config.sh` file exists in each TC directory.
- When `run_test.sh`, refer to the **tflite model** information in `config.sh`, download the file, and run the **tflite model** with the given **driver**.
- Related files : `run_test.sh` and `test_framework.sh`
- TC location :
    - `tests/scripts/framework/tests/` : Config directory for TC
    - `tests/scripts/framework/cache/` : TC (Downloaded tflite model files)

### Run tflite_run with various tflite models
- Driver : `tflite_run`
- Driver source location : `tools/tflite_run/`
- Usage :
```
$ ./tests/scripts/test-driver.sh \
    --artifactpath=. \
    --frameworktest
```
- Related pages : [tflite_run](https://github.sec.samsung.net/STAR/nnfw/tree/master/tools/tflite_run)

### Run nnapi_test with various tflite models
- `nnapi_test` runs tflite in two ways and compares the result:
    1. tflite interpreter
    2. `libneuralnetworks.so`, which could be PureACL or onert depending on linked to nnapi_test
- Driver : `nnapi_test`
- Driver source location : `tools/nnapi_test/`
- Usage :
```
$ ./tests/scripts/test-driver.sh \
    --artifactpath=. \
    --verification .
```


# nnpkg_test

`nnpkg_test` is a tool to run an nnpackage testcase.

`nnpackage testcase` is an nnpackage with additional data:

- input.h5 (input data)
- expected.h5 (expected outpute data)

`nnpkg_test` uses `nnpackage_run` internally to run `nnpackage`.

Then, it compares through `difftool` (either `i5diff` or `h5diff`).

`nnpkg_test` returns `0` on success, `non-zero` otherwise.

## Usage

```
$ tests/scripts/nnpkg_test.sh -h
Usage: nnpkg_test.sh [options] nnpackage_test
Run an nnpackage testcase

Returns
     0       success
  non-zero   failure

Options:
    -h   show this help
    -i   set input directory (default=.)
    -o   set output directory (default=.)
    -d   delete dumped file on failure.
         (dumped file are always deleted on success) (default=0)

Environment variables:
   nnpackage_run    path to nnpackage_run (default=Product/out/bin/nnpackage_run)
   difftool         path to i5diff or h5diff (default=h5diff)

Examples:
    nnpkg_test.sh Add_000                => run ./Add_000 and check output
    nnpkg_test.sh -i nnpkg-tcs Add_000   => run nnpkg-tcs/Add_000 and check output

```
