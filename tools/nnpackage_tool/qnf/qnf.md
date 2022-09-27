# qnf

`qnf` is a tool to convert between quantized and float.

It gets quantization parameters from input circle file.

## Prerequisite

$ python -r requirements.txt

## Usage

```
$ ./qnf.py -h
$ python tools/nnpackage_tool/qnf/qnf.py -h
usage: qnf.py [-h] [-o OUT_DIR] [-q | -d] h5 circle

positional arguments:
  h5                    path to h5 file either input or output to model
  circle                path to quantized circle model

optional arguments:
  -h, --help            show this help message and exit
  -o OUT_DIR, --output OUT_DIR
                        output directory
  -q, --quantize        quantize f32 to q8u using circle input's qparam
                        (default: false)
  -d, --dequantize      dequantize q8u to f32 using circle output's qparam
                        (default: false)

Examples:
  qnf.py -q input.h5 0c/0.circle    => generated quantized input as input_.h5
  qnf.py -d output.h5 0c/0.circle   => generated dequantized output as output_.h5
  qnf.py -o out/out.h5 -d output.h5 0c/0.circle   => generated dequantized output in out/output.h5
```
