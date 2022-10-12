### build for qemu
```bash 
arm-none-eabi-as -c -mthumb -mlittle-endian -march=armv7e-m -mcpu=cortex-m7 startup.S -o startup.o
arm-none-eabi-gcc -c -mthumb -ffreestanding -mlittle-endian -march=armv7e-m -mcpu=cortex-m7 test.c -o test.o
arm-none-eabi-ld -T test.ld test.o startup.o -o test.elf
```
### run with qemu
```bash 
qemu-system-arm -M netduinoplus2 -kernel test.elf -nographic
```
