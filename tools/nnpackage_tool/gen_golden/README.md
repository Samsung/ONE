# gen_golden

`gen_golden` is a tool to generate golden data from graph def (.pb) or tflite (.tflite).
It generates random inputs and run tensorflow, then save input and output in our h5 format.

## Prerequisite

Install tensorflow >= 1.12. It is tested with tensorflow 1.13, 1.14 and 2.0.

## Usage

```
$ ./gen_golden.py -h
usage: gen_golden.py [-h] [-o OUT_DIR] modelfile

positional arguments:
  modelfile             path to modelfile in either graph_def (.pb) or tflite (.tflite)

optional arguments:
  -h, --help            show this help message and exit
  -o OUT_DIR, --output OUT_DIR
                        output directory

Examples:
  gen_golden.py Add_000.pb              => generate input.h5 and expected.h5 in ./
  gen_golden.py -o ~/tmp Add_000.tflite => generate input.h5 and expected.h5 in ~/tmp/
```
