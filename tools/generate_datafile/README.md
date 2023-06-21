# Generate Datafile

This script aims to generate data files for onert_train tool.

This tool creates random input data automatically and it executes
onert_run tool to create output result data using the generated random input data.
It saves the input and output data as binary file format.

You can use `--num-runs` option to generate multiple data. This option combines
multiple input and output data and save them as a single input and output file.

You can also specify the input and output file name using `--input` and `--output`
options.

## How to use

### Prerequirement

- modelfile
- onert_run tool

### Basic usecase

Create one input and output datafile.

```
python3 ./tools/generate_datafile/generate_datafile.py \
  [onert_run tool path] \
  [modelfile path]
```

### Specify input and output file name

Create one input and output datafile with given name.

```
python3 ./tools/generate_datafile/generate_datafile.py \
  [onert_run tool path] \
  [modelfile path] \
  --input input \
  --output output
```

### Generate multiple data

Combine multiple test results and create one input and output datafile.

```
python3 ./tools/generate_datafile/generate_datafile.py \
  [onert_run tool path] \
  [modelfile path] \
  --num-runs [number]
```

## Example

### Create 50 data files using a.model

```
$ python3 ./tools/generate_datafile/generate_datafile.py \
  ./Product/out/bin/onert_run \
  ./test-models/a.model \
  --num-runs 50 \
  --input input.50 \
  --output output.50

Model Filename ./test-models/a.model
./test-models/a.input.10.0 is generated.
./test-models/a.output.10.0 is generated.
===================================
MODEL_LOAD   takes 1.914 ms
PREPARE      takes 5.744 ms
EXECUTE      takes 16.765 ms
- MEAN     :  16.765 ms
- MAX      :  16.765 ms
- MIN      :  16.765 ms
- GEOMEAN  :  16.765 ms
===================================
...
Generated input and output files
Done

$ ls -al ./test-models/a.*.bin
-rw-rw-r-- 1 jyoung jyoung  156800  6월 21 15:55 ./test-models/a.input.50.bin
-rw-rw-r-- 1 jyoung jyoung    2000  6월 21 15:55 ./test-models/a.output.50.bin
```
