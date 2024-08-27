# Training MobileNetV2 on ImageNet Dataset using ONERT

Let's train a MobileNetV2 model on ImageNet Dataset using ONERT.

## Prepare dataset

ImageNet dataset is an image database organized according to the WordNet hierarchy in which each node of hierarchy is depicted by hundreds and thousands of images. ImageNet contains more than 20,000 categories, with a typical category, such as "balloon" or "strawberry", consisting of several hundred images. There are serveral versions of ImageNet dataset in the TensorFlow datasets. Among them, ImageNet-A dataset is used in this document. ImageNet-A is a set of images labelled with ImageNet labels that were obtained by collecting new data and keeping only those images that ResNet-50 models fail to correctly classify. ONERT provides a `tf_dataset_converter` tool to download ImageNet-A dataset from TensorFlow Datasets and converts it to binary format for ONERT training.

```bash
$ python3 tools/generate_datafile/tf_dataset_converter/main.py \
--dataset-name imagenet_a \
--prefix-name imagenet_a \
--split test \
--length 100 \
--model mobilenetv2
Shape of images : (224, 224, 3)
Shape of labels: () <dtype: 'int64'>
class names[:10]  : ['n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475', 'n01496331', 'n01498041', 'n01514668', 'n01514859', 'n01518878']
class length      : 1000
The data files are created!
$ tree out
out
├── imagenet_a.test.input.100.bin
└── imagenet_a.test.output.100.bin

0 directories, 2 files
```

## Prepare model

Download MobileNetV2 model from TensorFlow Keras Application. This code generates a `model.tflite` TFLite model file.

```python
import tensorflow as tf

model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(224,224,3))

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

## Convert tflite model to circle model

ONERT can take a model from a tflite file or a circle file. So you can use this tflite model file directly. To use the circle model instead of the tflite model, you must convert it to the circle model using the 'onecc' tool, and 'onecc' can only be used in Ubuntu. When the ‘one-compiler’ debian package is installed on the [ONE release page](https://github.com/Samsung/ONE/releases), it is installed in the ‘/usr/share/one/bin’ directory. Please refer to the help for detailed guidelines. The following command converts the tflite model into a circle model.

```bash
$ onecc import tflite -- -i ./model.tflite -o ./model.circle
```

## Run training

Let's train the model using `onert_train`.

```bash
$ ./Product/out/bin/onert_train \
--epoch 5 \
--loss 2 \                      # categorical crossentropy
--loss_reduction_type 1 \       # sum over batch size
--optimizer 2 \                 # adam
--learning_rate 0.001 \
--batch_size 10 \
--num_of_trainable_ops -1 \     # train all operations
--load_input:raw ./out/imagenet_a.test.input.100.bin \
--load_expected:raw ./out/imagenet_a.test.output.100.bin \
model.circle
```

The result of training:
```bash
Model Filename model.circle
== training parameter ==
- learning_rate        = 0.001
- batch_size           = 10
- loss_info            = {loss = categorical crossentropy, reduction = sum over batch size}
- optimizer            = adam
- num_of_trainable_ops = -1
========================
Epoch 1/5 - time: 486.584ms/step - loss: [0] 7.2355
Epoch 2/5 - time: 483.868ms/step - loss: [0] 5.5251
Epoch 3/5 - time: 468.033ms/step - loss: [0] 4.6092
Epoch 4/5 - time: 481.408ms/step - loss: [0] 4.3135
Epoch 5/5 - time: 478.066ms/step - loss: [0] 4.2332
===================================
MODEL_LOAD   takes 14.1350 ms
PREPARE      takes 227.5800 ms
EXECUTE      takes 24018.8730 ms
- Epoch 1      takes 4865.8380 ms
- Epoch 2      takes 4838.6810 ms
- Epoch 3      takes 4680.3250 ms
- Epoch 4      takes 4814.0820 ms
- Epoch 5      takes 4780.6580 ms
===================================
```
