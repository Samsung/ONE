# Compute

`compute` directory is for the libraries for actual computation of neural network operations. These libraries are used by backends. Currently we have two libraries.

## ARMComputeEx

It is an extension of ARM [ComputeLibrary](https://github.com/ARM-software/ComputeLibrary), in order to support some operations that are not yet supported by ComputeLibrary. It is used by `acl_cl` and `acl_neon` backends.

The code structure looks just like ComputeLibrary's. Some of the code could be copied from the latest version of ComputeLibrary to support some operations quickly when those are not included in the latest version yet.

## cker

"cker" stands for Cpu KERnel. It is a port of Tensorflow lite's operation kernels with some additions. It is used by the `cpu` backend.
