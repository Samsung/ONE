# Training

## Overview

ONERT supports model training. In particular, it supports training not only in the Host environment but also in the On-device environment and provides features for on-device training. Training is an important part of developing and improving models. While general learning in the Host environment involves training with a large amount of data using many resources, in the on-device environment there is limited space to store data, and the resources that can be used are also limited. Therefore, efficient learning using less data and small resources is required for on-devcie training. ONERT provides various features to meet these requirements.

It supports on-device training based on existing trained models and supports transfer learning to learn new data added to the trained model. It also provides supervised fine-tuning to improve the accuracy of the trained model. This allows ONERT to support efficient learning with less data.

In order to efficiently learn models with limited resources, ONERT not only uses less memory by reusing it memory space during training but also quickly improves accuracy of trained model by performing optimized kernel for each operation.

Trained model can be saved to be deployed using ONERT API. This allows users to redeploy it to other environments or real applications to perform inference and retraining.

## Training in ONERT

### Training options

ONERT supports the following options:

- Loss function
  - w/ loss reduction type
- Optimizer
  - w/ learning rate
- Batch size
- Num of trainable operations

### Training process

Prerequisites:

- Prepare a circle model to be trained.
- Prepare your dataset and preprocess it if necessary.

Training process in ONERT consists of the following steps:

1. Create a session.
2. Load a circle model.
   - (optional) Load the checkpoint. (for fine-tuning)
3. Set training information.
4. Prepare training a model.
5. Set input and expected data.
6. Run training.
   - (optional) Get loss and accuracy.
   - (optional) Save the checkpoint.
7. Validate the model.
8. Export the trained model for inference.

### Training tools

- [onert_train](runtime/tests/tools/onert_train): A tool to train neural networks with ONERT.
- [generate_datafile](tools/generate_datafile): A tool to generate data files for ONERT training.
- [circle_plus_gen](tools/circle_plus_gen): A tool to generate Circle+ model from circle model.

## Example

### Training well-known models using ONERT

- [Training a simple CNN model on MNIST Dataset using ONERT](docs/runtime/training_cnn_on_mnist.md)
- [Training MobileNetV2 model on ImageNet Dataset using ONERT](docs/runtime/training_mobilenetv2_on_imagenet.md)

### Advanced Topics

- [Transfer Learning](docs/runtime/transfer_learning.md)
