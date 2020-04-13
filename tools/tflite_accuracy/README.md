Using the batch execution tool
==============================

The batch execution tool (`tflite_accuracy`) can be used to run experiments
where execution time and accuracy are to be measured on a test set.
`tflite_accuracy` reads a neural network model from a file and a series of
input images from a directory, runs each image through the network,
and collect statistics, such as execution time and accuracy.

In order to run this tool, you'll need:

* a model in `.tflite` format;
* a set of preprocessed input images in binary format, properly named
(see below).

`tflite_accuracy` expects all the input images to be located in the same directory
in the file system. Each image file is the binary dump of the network's
input tensor. So, if the network's input tensor is a `float32` tensor of 
format (1, 224, 224, 3) containing 1 image of height 224, width 224, and
3 channels, each image file is expected to be a series of 224 * 224 * 3 
`float32` values.

`tflite_accuracy` does **not** perform any preprocessing on the input tensor
(e.g., subtraction of mean or division by standard deviation). Each image 
file is treated as the final value of the input tensor, so all the
necessary preprocessing should be done prior to invoking the tool.

In order to calculate accuracy on the image set, `tflite_accuracy` needs to know
the correct label corresponding to each image. This information is
extracted from the file's name: the first four characters in the name are
assumed to be the numerical code of the image's class. So, a file named
`0123_0123456789.bin` is assumed to represent an image belonging to class
`123`. The remainder of the name (`0123456789` in the example) is assumed 
to be an identifier of the image itself.

The width and height each image can be informed via the command line
argument `--imgsize`, whose default value is 224.
