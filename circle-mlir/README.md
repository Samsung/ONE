# circle-mlir

Circle MLIR dialect and tools

## Tools provided

_onnx2circle_
- conversion tool of ONNX to Circle model for `compiler`

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
sudo apt-get install autoconf automake libtool unzip wget
sudo apt-get install devscripts debmake debhelper lcov
sudo apt-get install python3 python3-pip python3-venv python3-dev python3-all dh-python
```

Install `one-compiler` package;
- download and install latest `ONE Release` from https://github.com/Samsung/ONE/releases
- actually we only need `circle-interperter` tool for validation, but not ready yet

Or use local build/install version of `ONE/compiler` and set environment variable;
```
export ONE_COMPILER_ROOT=/home/user/one/build/install
```

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

NOTE `llvm-project` is built as `Debug` which may require 32G or more RAM.
- if build fails for some reason, please change to `-DCMAKE_BUILD_TYPE=Release`
  `prep` target in `Makefile.sample` file.

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

## TensorFlow source code

Some source codes are referenced from TensorFlow and the file path is added to
inside our source.

Current codes are from `v2.12.1` tag.
