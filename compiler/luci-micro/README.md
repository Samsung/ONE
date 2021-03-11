0. install  mbed-cli https://os.mbed.com/docs/mbed-os/v6.8/build-tools/install-and-set-up.html
1. ```bash
mbed deploy
```
2. ```bash
mbed compile --source . -m DISCO_F746NG -t GCC_ARM --profile release.json --flash
```