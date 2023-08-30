# tf dataset converter

## What is tf dataset converter?

_tf dataset converter_ is a tool which converts tensorflow datasets to datasets for `onert_train`.

## Possible datasets
- Tensorflow datasets with [ClassLabel feature](https://www.tensorflow.org/datasets/api_docs/python/tfds/features/ClassLabel)

## Prerequisite
- Python 3.8 (python3.8, python3.8-dev packages)
- Python packages required

## Usage
usage: main.py [-h] [-s] [-d Dataset] [-o Dir] [-p Prefix] [-l N] [-t N]

Convert a dataset of tensorflow to onert format

options:
  -h, --help            show this help message and exit
  -s, --show-datasets   show dataset list
  -d Dataset, --dataset-name Dataset
                        name of dataset to be converted (default: "fashion_mnist")
  -o Dir, --out-dir Dir
                        relative path of the files to be created (default: "out")
  -p Prefix, --prefix-name Prefix
                        prefix name of the file to be created (default: "")
  -l N, --train-length N
                        Number of data for training (default: 1000)
  -t N, --test-length N
                        Number of data for training (default: 100)

## Example
### Install required packages
```
$ python3 -m pip install -r requirements.txt
```

### Show dataset list
```
$ python3 main.py --show-datasets
Dataset list :
[abstract_reasoning,
accentdb,
...
fashion_mnist,
...
robotics:mt_opt_sd]
```

### Convert dataset to onert format
```
$ python3 main.py \
 --dataset-name fashion_mnist \
 --prefix-name fashion-mnist \
 --train-length 2000 \
 --test-length 200
```
```
$ tree out
out
├── fashion-mnist.test.input.200.bin
├── fashion-mnist.test.output.200.bin
├── fashion-mnist.train.input.2000.bin
└── fashion-mnist.train.output.2000.bin
```
