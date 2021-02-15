## MicroLuci

### How to build this project&

0. Install python 3.8+ version

1. Install mbed-cli=2.0:

```bash
pip install mbed-tools

```

2. Install dependences with *mbed-os*:

```bash
sudo apt-get install python3-dev
pip install ninja
pip install pyelftools
mbed-tools deploy

```


3. Find your gcc-arm toolchain (install from: https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-rm/downloads) and add it to PATH.

4. Import libs:
```bash
python import_libs.py

```

5. Compile project for your board (add `-f` option for push firmware to board):

```bash
mbed-tools compile -m DISCO_F746NG -t GCC_ARM -c

```

### Usefull information

- debugger for clion https://habr.com/ru/post/345670/
- mbed-cli example https://os.mbed.com/docs/mbed-os/v6.6/quick-start/compiling-the-code.html
- openocd http://openocd.org/getting-openocd/

## Docker
```bash
sudo docker pull slavikmipt/one-mbed-env:latest
sudo docker run -it --privileged -v /dev/disk/by-id:/dev/disk/by-id -v /dev/serial/by-id:/dev/serial/by-id slavikmipt/one-mbed-env:latest
```
