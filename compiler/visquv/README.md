# visquv

_visquv_ is a tool to view _visq_ generated quantization error files with
web browser, using nodejs.

## Test

Build and local test install
```
NNCC_WORKSPACE=build/debug ./nncc configure -DCMAKE_INSTALL_PREFIX="build/debug.install"
NNCC_WORKSPACE=build/debug ./nncc build install
```

Prerequisite to test run
```
sudo apt-get install nodejs
```

Test run
```
cd build/debug.install/bin
./one-visquv visquv/test_error.json
```

Message would be something like this.
```
Available IP: a.b.c.d
Listening: a.b.c.d:7070
Open in browser with http://a.b.c.d:7070/
```

Open `http://a.b.c.d:7070/` in browser with `a.b.c.d` to real address.
