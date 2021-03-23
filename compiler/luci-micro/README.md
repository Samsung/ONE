This is benchmarks for luci-micro, tflite-micro using mbed-cli 1

install mbed-cli https://os.mbed.com/docs/mbed-os/v6.8/build-tools/install-and-set-up.html

```bash
mbed deploy
```

```bash
python import_libs.py
```

```bash
mbed compile --source . -m DISCO_F746NG -t GCC_ARM --profile release.json --flash
```
