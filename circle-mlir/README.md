# circle-mlir

Circle MLIR dialect and tools

## Tools provided

_onnx2circle_
- conversion tool of ONNX to Circle model for `compiler`
- to replace not-maintained-anymore onnx-tensorflow package

## How to build

Use provided `Makefile.sample` or create your own `Makefile`
```
ln -s Makefile.sample Makefile
```
- `Makefile` is in `.gitignore` to let developers use own Makefile.

### Prerequisite

```
sudo apt-get install build-essential cmake git fakeroot
sudo apt-get install autoconf automake libtool unzip wget
sudo apt-get install devscripts debmake debhelper lcov
sudo apt-get install python3 python3-pip python3-venv python3-dev python3-all dh-python

python3 -m pip install --upgrade pip setuptools
python3 -m pip install numpy h5py==3.8.0
```

### Prepare externals

### Debug build

Prepare overlay
```
make overlay
```

Build submodules in venv
```
source infra/overlay/venv/bin/activate
make prep
```
NOTE `llvm-project` is built as `Debug` which may require 32G or more RAM.
- if build fails for some reason, please change back to
  `-DCMAKE_BUILD_TYPE=Release` in `prep:` target in `Makefile.sample` file.
- build and test needs venv python packages.

NOTE `overlay` and `submodules` builds are needed only once.

Configure and build
```
make cfg
make debug
```

Test build
```
make test
```
- optionally, set `ONE_COMPILER_ROOT` to alternate PATH for local ONE build
  ```
  ONE_COMPILER_ROOT=/home/user/one/build/install make test
  ```

To clean up existing build results
```
make clean
```

To clean up also `overlay` and `submodules`
```
make cleanall
```
- NOTE when using `CIRCLE_MLIR_LOCALINST`, need to manually clean up this folder

### Release build

Release build is available as follows.
Others not mentioned are same as above Debug build.

Build submodules in venv
```
source infra/overlay/venv/bin/activate
make prepr
deactivate
```

Configure and build
```
make cfgr
make rel
```

Test build
```
make testr
```

### Test coverage

To get test coverage report, run as following commands.
- assume you already have done `make overlay` and `make prepcov`
- you can skip `make prepcov` step if you are using local installation with `CIRCLE_MLIR_LOCALINST`
- or you can reuse `CIRCLE_MLIR_LOCALINST` for existing debug or release build submodules with 
`cfgcov` target such as `CIRCLE_MLIR_LOCALINST=$(pwd)/build/debug/submodules make cfgcov`
```
source infra/overlay/venv/bin/activate
make cfgcov
deactivate

make debugcov
make testcov
make gencov
```

Open `converage/html/index.html` file in web browser to see the reports.

To generate from second run and so on in your local machine, you will have to
remove existing files before running `gencov`
```
rm -rf coverage
make gencov
```

To run this with Docker image, use `cfgcovdi` target instead of `cfgcov`.
```
make cfgcovdi
make debugcov
make testcov
make gencov
```


## Local format check

Install prerequiste package.
```
sudo apt-get install clang-format-12 python3 python3-pip
python3 -m pip install yapf==0.32.0
```

Run format checker.
```
bash ./infra/tools/format
```
or with `Makefile` from `Makefile.sample`
```
make format
```

## Dump debug logs

To see logs during conversion with `onnx2circle` tool, set `CM_PASS_DUMP=1` for
preprocessing ONNX and ONNX to circle conversion, or set `CM_PASS_DUMP=2` to see
additional logs for circle rewrite.

```
CM_PASS_DUMP=2 onnx2circle input.onnx output.circle
```

You can give `-debug` option to see general MLIR logs or `-debug-only=o2c`
option to see only logs from onnx2circle.

```
onnx2circle -debug-only=o2c input.onnx output.circle
```

## TensorFlow source code

Some source codes are referenced from TensorFlow and the file path is added to
inside our source.

Current codes are from `v2.12.1` tag.
