# here I write how I run model on my computer

## goal of this script

Here the author has attempted to implement a program capable of running any of the 4 models (caffe, caffe2, tflite, onnx) in a simple and user-friendly manner. The goal of the program is to get the file containing the output of the computation graph at the program output.

## examples of code running in author's local machine
The purpose of the examples below is to demonstrate which arguments and in which order you should use to run this script correctly.

caffe:
```
$ python3 model_runner.py -m  caffe1_runer/inception-v3_ref.caffemodel  caffe1_runer/inception-v3_ref.prototxt  -i caffe1_runer/ILSVRC2012_val_00000002.JPEG.tfl.hdf5
```
caffe2:
```
$ python model_runner.py -m  caffe2_runer_and_photo/caffe2_models/init_net.pb  caffe2_runer_and_photo/caffe2_models/predict_net.pb -i randomInput.hdf5
```
tflite:
```
$ python model_runner.py -m  tflite_runer_and_photo/TST-1-2\ AVARAGE_POOP_2D.tflite -i  tflite_runer_and_photo/in.hdf5
```
onnx:
```
$ python model_runner.py  -m onnx_runer/model.onnx -i RANDOM.hdf5
```

 ------
 
 ## parameters and short comment
 
 -m mean pre learned model which you run
 -i mean model's input

