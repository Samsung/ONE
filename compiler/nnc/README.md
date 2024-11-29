# nnc
Neural Network Compiler

### DESCRIPTION

nnc is a neural network compiler that transforms neural networks of various formats into source or machine code.
> At this moment, only two NN are supported (MobileNet and InceptionV3) in Tensorflow Lite or Caffe format.

### SYNOPSIS

nnc OPTIONS

### OPTIONS

        --help, -h            -    print usage and exit
        --caffe               -    treat input file as Caffe model
        --tflite              -    treat input file as Tensor Flow Lite model
        --target              -    select target language to emit for given architecture.
                                   Valid values are 'x86-c++', 'interpreter'
        --nnmodel, -m         -    specify input file with NN model
        --output, -o          -    specify name for output files
        --output-dir, -d      -    specify directory for output files
        --input-model-data    -    interpreter option: specify file with neural network input data.
                                   This file contains array of floats in binary form
        --input-node          -    interpreter option: set input node in Computational Graph
        --output-node         -    interpreter option: set output node in Computational Graph



### USAGE

Assuming that user has already installed nnc as follows:
```
$ cmake <path_to_nnc_sources> -DCMAKE_INSTALL_PREFIX=<path_to_install>
$ make all && make install
```

Also assuming that we have tflite model (for example inceptionv3.tflite)

**1. Running nnc in interpreter mode:**
```
<path_to_install>/bin/nnc \
--nnmodel inceptionv3.tflite \
--target interpreter \
--input-model-data data.file \
--input-node input --output-node output
```

**2. Running to generate C/C++ source code:**

```
<path_to_install>/bin/nnc \
--nnmodel inceptionv3.tflite \
--target x86-c++ \
--output inception \
--output-dir output_dir
```

