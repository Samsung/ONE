[![GitHub release (latest
SemVer)](https://img.shields.io/github/v/release/Samsung/ONE)](https://github.com/Samsung/ONE/releases)
[![Documentation Status](https://readthedocs.org/projects/nnfw/badge/?version=latest)](https://nnfw.readthedocs.io/en/latest/?badge=latest)
![GitHub commit activity](https://img.shields.io/github/commit-activity/w/Samsung/ONE?color=light%20green)
[![Gitter](https://img.shields.io/gitter/room/Samsung/ONE?color=orange)](https://gitter.im/Samsung/ONE)

# **ONE** (On-device Neural Engine)

<img src='docs/images/logo_original_samsungblue_cropped.png' alt='ONE Logo' width='400' />

A high-performance, on-device neural network inference framework.

## Goal

This project **ONE** aims at providing a high-performance, on-device neural network (NN) inference
framework that performs inference of a given NN model on processors, such as CPU, GPU, DSP or NPU.

We develop a runtime that runs on a Linux kernel-based OS platform such as Ubuntu, Tizen, or 
Android, and a compiler toolchain to support NN models created using various NN training frameworks such 
as Tensorflow or PyTorch in a unified form at runtime.

## Overview

- [Background](docs/overview/background.md)
- [Roadmap](docs/overview/roadmap.md)

## Getting started

- For the contribution, please refer to our [contribution guide](docs/howto/how-to-contribute.md).
- You can also find various how-to documents [here](docs/howto).

## Feature Request

You can suggest development of **ONE**'s features that are not yet available.

The functions requested so far can be checked in the [popular feature request](https://github.com/Samsung/ONE/issues?q=label%3AFEATURE_REQUEST+) list.

- If the feature you want is on the list, :+1: to the body of the issue. The feature with the most
:+1: is placed at the top of the list. When adding new features, we will prioritize them with this reference.
Of course, it is good to add an additional comment which describes your request in detail.

- For features not listed, [create a new issue](https://github.com/Samsung/ONE/issues/new).
Sooner or later, the maintainer will tag the `FEATURE_REQUEST` label and appear on the list.

We expect one of the most frequent feature requests would be the operator kernel implementation.
It is good to make a request, but it is better if you contribute by yourself. See the following guide,
[How to add a new operation](docs/howto/how-to-add-a-new-operation.md), for help.

We are looking forward to your participation.
Thank you in advance!

## How to Contact

- Please post questions, issues, or suggestions into [Issues](https://github.com/Samsung/ONE/issues). This is the best way to communicate with the developer.
- You can also have an open discussion with community members through [gitter.im](https://gitter.im/Samsung/ONE) channel.
