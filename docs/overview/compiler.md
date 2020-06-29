# Introduction

This document discusses the ONE compiler. ONE compiler quantizes and optimizes the input model. It takes many model formats and converts to a common [Circle](#circle-format) format which is then deployed.

# Contents

* [Build Setup](#Build-Setup)
* [Supported Formats](#Supported-Formats)
* [Circle Format](#Circle-Format)

## Build Setup

* System Setup   
This setup is to build the ONE compiler on Ubuntu. Its tested on Ubuntu 16.04 as well as Ubuntu 18.04.  
Get the code [Samsung ONE](https://github.com/Samsung/ONE) from Github. 

* Dependencies
    * Cmake (Greater than or equal to 3.1)
    * g++ toolchain (Greater than or equal to 6)
    * python virtual environment (python3-venv for python3)
    * HDF5 library (libhdf5-dev)

* Configure  
```./nncc configure```

You can pass CMAKE flags to above configuration command like (-DENABLE_TEST=1).  
This would spit the missing packages if any errors.  

* Build  
```./nncc build```

After successful build, circle files are created inside build/compiler/**/*.circle.

* Test  
```./nncc test```  

This for now runs a suite of tests.  

## Supported Formats

Neural Networks can be held in many formats & ONE supports muliple formats too.  
But the primary focus for ONE is the TensorFlow Lite format.  

## Circle Format

Find Flatbuffers circle format [here](https://github.com/Samsung/ONE/blob/master/nnpackage/schema/circle_schema.fbs).
