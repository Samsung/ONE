- install python 3.8+ version

- install https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-rm/downloads

- install mbed cli 2 https://os.mbed.com/docs/mbed-os/v6.6/build-tools/install-or-upgrade.html
```bash
mbed-tools deploy
python import_libs.py
mbed-tools compile -m DISCO_F746NG -t GCC_ARM
```
- debugger for clion https://habr.com/ru/post/345670/
- mbed-cli example https://os.mbed.com/docs/mbed-os/v6.6/quick-start/compiling-the-code.html
- openocd http://openocd.org/getting-openocd/

## Docker
- sudo docker pull slavikmipt/one-mbed-env:latest
- sudo docker run -it --privileged -v /dev/disk/by-id:/dev/disk/by-id -v /dev/serial/by-id:/dev/serial/by-id slavikmipt/one-mbed-env:latest
