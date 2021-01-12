- install https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-rm/downloads

- install mbed cli 2 https://os.mbed.com/docs/mbed-os/v6.6/build-tools/install-or-upgrade.html
```bash
mbed-tools deploy
python import_libs.py
mbed-tools compile -m DISCO_F746NG -t GCC_ARM
```