# Training a simple CNN model on MNIST using ONERT

Let's train a simple neural network on MNIST dataset using ONERT.

## Prepare dataset

MNIST dataset consists of 60,000 training images and 10,000 test images. Each image is a 28x28 grayscale image, associated with a label from 10 classes. This dataset is available in TensorFlow datasets. ONERT provides a `tf_dataset_converter` tool to download dataset from TensorFlow Datasets and converts it to binary format for ONERT training.

```bash
$ python3 tools/generate_datafile/tf_dataset_converter/main.py \
--dataset-name fashion_mnist --prefix-name fashion-mnist --model mnist
Shape of images : (28, 28, 1)
Shape of labels: () <dtype: 'int64'>
class names[:10]  : ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
class length      : 10
The data files are created!
$ tree out
out
├── fashion-mnist.test.input.100.bin
├── fashion-mnist.test.output.100.bin
├── fashion-mnist.train.input.1000.bin
└── fashion-mnist.train.output.1000.bin

0 directories, 4 files
```

## Prepare model

Prepare a simple neural network for trainng MNIST datasets using the `tf.keras` API. This code generates a `model.tflite` TFLite model file.

```python
import tensorflow as tf
import tensorflow_datasets as tfds

# Create a model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Donwload train dataset
ds = tfds.load('mnist', split='train', as_supervised=True)
ds = ds.batch(128)

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(ds, epochs=5)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

## Convert tflite model to circle model

ONERT takes both the tflite model and the circle model. So you can use this tflite model file directly. To use the circle model instead of the tflite model, you have to convert it to the circle model using the 'onecc' tool, and 'onecc' can only be used in Ubuntu. When the ‘one-compiler’ debian package is installed on the ONE release page, it is installed in the ‘/usr/share/one/bin’ directory. Please refer to the help for detailed guidelines. The following command converts the tflite model into a circle model.

```bash
$ onecc import tflite -- -i ./model.tflite -o ./model.circle
```

## Run training

Let's train the model using `onert_train`.

```bash
$ ./Product/out/bin/onert_train \
--epoch 5 \
--loss 1 \                    # mean squared error
--loss_reduction_type 1 \     # sum over batch size
--optimizer 2 \               # adam
--learning_rate 0.001 \
--batch_size 10 \
--num_of_trainable_ops -1 \   # train all operations
--load_input:raw ./out/fashion-mnist.train.input.1000.bin \
--load_expected:raw ./out/fashion-mnist.train.output.1000.bin \
model.circle
```

The result of training:
```bash
Model Filename model.circle
== training parameter ==
- learning_rate        = 0.001
- batch_size           = 10
- loss_info            = {loss = mean squared error, reduction = sum over batch size}
- optimizer            = adam
- num_of_trainable_ops = -1
========================
Epoch 1/5 - time: 1.602ms/step - loss: [0] 0.1082
Epoch 2/5 - time: 1.674ms/step - loss: [0] 0.0758
Epoch 3/5 - time: 1.624ms/step - loss: [0] 0.0611
Epoch 4/5 - time: 1.624ms/step - loss: [0] 0.0549
Epoch 5/5 - time: 1.635ms/step - loss: [0] 0.0516
===================================
MODEL_LOAD   takes 0.2440 ms
PREPARE      takes 1.7130 ms
EXECUTE      takes 829.2350 ms
- Epoch 1      takes 160.1870 ms
- Epoch 2      takes 167.4430 ms
- Epoch 3      takes 162.4110 ms
- Epoch 4      takes 162.4300 ms
- Epoch 5      takes 163.5280 ms
===================================
```
