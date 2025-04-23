# circle-mlir

Circle MLIR dialect and tools

## Provided Tools

_onnx2circle_
- a tool that converts ONNX models to Circle format, used by `compiler`

## How to build

Use provided `Makefile.sample` or create your own `Makefile`;
```
ln -s Makefile.sample Makefile
```
- `Makefile` is in `.gitignore` to let developers use own Makefile.

### Prerequisite

Install necessary packages;
```
sudo apt-get install build-essential cmake git
sudo apt-get install autoconf automake libtool pkg-config unzip wget libhdf5-dev
sudo apt-get install devscripts debmake debhelper lcov
sudo apt-get install python3 python3-pip python3-venv python3-dev python3-all dh-python
```

Install `one-compiler` package;
- download and install latest `ONE Release` from https://github.com/Samsung/ONE/releases
- actually we only need `circle-interperter` tool for validation, but not ready yet

Alternatively, if you have a locally built version of `compiler`,
set the environment variable as follows;
```
export ONE_COMPILER_ROOT=/home/user/one/build/install
```

### CIRCLE_MLIR_LOCALINST environment (optional)

For working with multiple clones, each clone may have externals build binaries,
which may take about 150GB to 200GB of disk space.

To overcome this issue, use `CIRCLE_MLIR_LOCALINST` and `CIRCLE_MLIR_LOCALINST_USE`
environment variable to build a single binary and use for other clones.

For example, to build externals in first clone,
```
export CIRCLE_MLIR_LOCALINST=$HOME/local/circlemlir
make prep
```

and for other clones that uses externals build from the first clone,
```
export CIRCLE_MLIR_LOCALINST=$HOME/local/circlemlir
export CIRCLE_MLIR_LOCALINST_USE
make prep
```
NOTE, `onnx-mlir` is cloned in `externals` folder, as `onnx-mlir` source is
referenced from `circle-mlir`.

### Prepare overlay

This will prepare virtual-env with necessary python packages installed;
```
make overlay
source infra/overlay/venv/bin/activate
```

NOTE `overlay` build is needed only once.

### Debug build

Build externals;
```
make prep
```

NOTE `llvm-project` builds in `Debug` mode, which may require 32GB or more RAM.
- if the build fails, try changing the build type to `-DCMAKE_BUILD_TYPE=Release`
  in the `prep` target of `Makefile.sample` file.

NOTE `prep` build is needed only once.

Configure and build;
```
make cfg
make debug
```

Test build;
```
make test
```

To clean up existing build results;
```
make clean
```

To clean up also `overlay` and `submodules`;
```
make cleanall
```

### Release build

Release build is almost same as Debug build.

Build externals;
```
make prepr
```

Configure and build;
```
make cfgr
make rel
```

Test build
```
make testr
```

Install tools into `build/install`
```
make install
```

### Docker build

You can build within Docker image, with pre-built `externals`.

Pull Docker image
```
docker pull nnfw/circle-mlir-build:jammy
```
NOTE there is only Ubuntu22.04 version, as of writing this.

Enter shell
```
docker run \
--rm \
-u $(id -u):$(id -g) \
-it \
-v $HOME:/home/circlemlir \
-w /home/circlemlir \
nnfw/circle-mlir-build:jammy
```

Inside the Docker image shell, cd to `circle-mlir`
```
cd circle-mlir
```

For debug build
```
make cfgdi
make debug
make test
```

For Release build
```
make cfgri
make rel
make testr
```

### Test coverage

To get test coverage report, run as following commands.
- assume you already have done `make overlay` and `make prepcov`
- you can skip `make prepcov` step if you are using local installation with `CIRCLE_MLIR_LOCALINST`
```
make cfgcov

make debugcov
make testcov
make gencov
```

Open `converage/html/index.html` file in web browser to see the reports.

To generate from second run and so on in your local machine, you need to
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

## Dump debug logs

To see logs during conversion with `onnx2circle` tool, set `CM_PASS_DUMP=1` for
preprocessing ONNX and ONNX to circle conversion, or set `CM_PASS_DUMP=2` to see
additional logs for circle rewrite.

```
CM_PASS_DUMP=2 onnx2circle input.onnx output.circle
```

You can give `-debug` option to see all general MLIR logs or `-debug-only=o2c`
option to see only logs from onnx2circle.

```
onnx2circle -debug-only=o2c input.onnx output.circle
```

## TensorFlow source code

Some source codes are referenced from TensorFlow and the file path is added to
inside our source.

Current codes are from `v2.12.1` tag.
