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

You could train your model using the command like below.  

```bash
onert_train \
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
