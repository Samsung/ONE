# onert_train

`onert_train` aims to train ai model. This tool trains given the ai model entered by the user using a given input and an  expected output, and stores or inference the trained model.

The input models that can be supported by this tool are as follows.
- circle
- nnpackage

## Prerequisites

Required software tools:
  - libhdf5-dev
  - libboost-program-options-dev

```
sudo apt install -y libhdf5-dev libboost-program-options-dev
```

## Usage

`onert_train` takes a model(*.circle or nnpackage), training data files and training parameters.  

You could train your model using the command like below.  
```bash
$ onert_train \
--path [circle file or nnpackage] \
--load_input:raw [training input data] \
--load_expected:raw [training output data] \
--batch_size 32 \ 
--epoch 5 \
--optimizer 1 \             # sgd
--learning_rate 0.01 \   
--loss 2 \                  # cateogrical crossentropy
--loss_reduction_type 1     # sum over batch size
```
`onert_train --help` would help you to set each parameter.

## Example

To deliver quick insight to use `onert_train`, let's train a simple [mnist](https://www.kaggle.com/code/amyjang/tensorflow-mnist-cnn-tutorial) model. 

Before using `onert_train`, data files and a model file have to be ready.


### Prepare training data

The `onert_train` expects that a preprocessed dataset is given as binary files. <br/>
For convenience, we provide a tool([tf dataset convert](https://github.com/Samsung/ONE/tree/master/tools/generate_datafile/tf_dataset_converter)) that preprocesses tensorflow dataset and save it as binary files.

Let's convert mnist datasets to input binary files. 
```bash
# Move to tf_dataset_convert directory 
$ cd ONE/tools/generate_datafile/tf_dataset_converter

# install prerequisites
$ pip3 install -r requirements.txt

# generate binary data files
$ python3 main.py \ 
--dataset-name mnist \ 
--prefix-name mnist \ 
--model mnist 

# check data files are generated
# There are mnist.train.input.1000.bin, mnist.train.output.1000.bin
$ tree out
```

### Prepare a circle model file

`onert_train` gets a `*.circle` file or a nnpackage as input. <br/>
You could convert tf/tflite/onnx model file into circle file using [`onecc`](https://github.com/Samsung/ONE/tree/master/compiler/one-cmds). 

For example, if you start with tensorflow code like [here](https://github.com/Samsung/ONE/tree/master/tools/generate_datafile/tf_dataset_converter), you could save the model to tensorflow saved format by adding below code. 
```python 
model.save('mnist_model')
``` 

After that, you could convert the tensorflow model to circle using under command.
```bash 
$ onecc import tf --v2 --saved_model -i mnist_model -o mnist.circle
```
<!--TODO : Add How to inject training parameter into the circle model -->

### Run onert_train
Now you're ready to run `onert_train`.

Please pass your model file to `--modelfile` and data files to `--load_input:raw` and `--load_expected:raw`. <br/>
Also, don't forget to set training parameters according to each option. 

```bash 
$ onert_train \
--modelfile mnist.circle \
--load_input:raw mnist.train.input.1000.bin \
--load_expected:raw mnist.train.output.1000.bin \
--batch_size 32 \ 
--epoch 5 \
--optimizer 2 \          # adam
--learning_rate 0.001 \
--loss 2 \               # cateogrical crossentropy
--loss_reduction_type 1  # sum over batch size
```
