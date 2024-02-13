# Roadmap

This document describes the roadmap of the **ONE** project.

Project **ONE** aims at providing a high-performance, on-device neural network (NN) inference
framework that performs inference of a given NN model on processors, such as CPU, GPU, DSP, or NPU, 
on the target platform, such as Tizen, Android and Ubuntu. 

## Progress

Until last year, we already saw significant gains in accelerating with a single CPU or GPU backend.
We have seen better performance improvements, not only when using a single backend, but even when
mixing CPUs or GPUs, considering the characteristics of individual operations. It could give us an
opportunity to have a high degree of freedom in terms of operator coverage and possibly provide
better performance compared to single backend acceleration.

On the other hand, we introduced the compiler as a front-end. This will support a variety of deep
learning frameworks in relatively spacious host PC environments, while the runtime running on the
target device is intended to take a smaller burden. In this approach, the compiler and the runtime
will effectively share information among themselves by the Common IR, named _circle_, and a
container format which is referred to as the _NN Package_.

## Goal

In the meantime, we have been working on improving the acceleration performance based on the vision
model. From this year, now we start working on the voice model. The runtime requirements for the
voice model will be different from those of the vision model. There will be new requirements that
we do not recognize yet, along with some already recognized elements such as control flow and
dynamic tensor. In addition, recent studies on voice models require efficient support for specific
architectures such as attention, transformer and BERT. Also, depending on the characteristics of
most voice models with large memory bandwidth, we will have to put more effort into optimizing the
memory bandwidth at runtime.

## Deliverables

- Runtime
  + Control Flow support (IF/WHILE)
  + Dynamic Tensor support
  + High quality kernel development for UINT8 quantized model
  + End-to-end performance optimization for voice models
- Compiler
  + More than 100 operations support
  + Standalone _circle_ interpreter
  + Completion and application of _circle2circle_ pass
    - _circle-quantizer_ for UINT8 and INT16
    - _circle-optimizer_
  + Graphical _circle_ model viewer

## Milestones

- [2020 Project Milestones](https://github.com/Samsung/ONE/projects/1)

## Workgroups (WGs)

- We organize WGs for major topics and each WG will be working on its own major topic by breaking
  it into small tasks/issues, performing them inside WG and collaborating between WGs.
- The WG information can be found [here](workgroup.md).

