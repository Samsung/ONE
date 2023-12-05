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

optional arguments:
  -h, --help            show this help message and exit
  -s, --show-datasets   show dataset list
  -d Dataset, --dataset-name Dataset
                        name of dataset to be converted (default: "fashion_mnist")
  -o Dir, --out-dir Dir
                        relative path of the files to be created (default: "out")
  -p Prefix, --prefix-name Prefix
                        prefix name of the file to be created (default: "")
  --split [Split [Split ...]]
                        Which split of the data to load (default: "train, test")
  --length [N [N ...]]  Data number for items described in split (default: "1000, 100")
  -m Model, --model Model
                        Model name to use generated data (default: mnist)

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

Convert `fashion_mnist` dataset for use in `mnist` model
```
$ python3 main.py \
 --dataset-name fashion_mnist \
 --prefix-name fashion-mnist \
 --model mnist
Shape of images : (28, 28, 1)
Shape of labels: () <dtype: 'int64'>
class names[:10]  : ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
class length      : 10
The data files are created!
```
```
$ tree out
out
├── fashion-mnist.test.input.100.bin
├── fashion-mnist.test.output.100.bin
├── fashion-mnist.train.input.1000.bin
└── fashion-mnist.train.output.1000.bin
```

Convert `imagenet_a` dataset for use in `MobileNetV2` model
```
$ python3 main.py \
 --dataset-name imagenet_a \
 --prefix-name imagenet_a \
 --split test \
 --length 1000 \
 --model mobilenetv2
Shape of images : (224, 224, 3)
Shape of labels: () <dtype: 'int64'>
class names[:10]  : ['n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475', 'n01496331', 'n01498041', 'n01514668', 'n01514859', 'n01518878']
class length      : 1000
The data files are created!
```
```
$ tree out
out
├── imagenet_a.test.input.1000.bin
└── imagenet_a.test.output.1000.bin
```
