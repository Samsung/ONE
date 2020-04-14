[![Inner Source](https://img.shields.io/badge/innersource-incubating-orange)](http://mosaic.sec.samsung.net/kms/comty.do?comtyId=279217135&menuId=324419756&postId=373704030&page=view&type=LIST)

# nnfw

A high-performance, on-device neural network inference framework

## Goal

This project _nnfw_ aims at providing a high-performance, on-device neural network (NN) inference
framework that performs inference of a given NN model on processors, such as CPU, GPU, or NPU, in
the target platform, such as the Linux kernel based OS including Tizen.

## Project Documents

- [Roadmap](docs/nnfw/roadmap.md)
- [SW Requirement Specification](docs/nnfw/project/2019_requirement_specification.md)
- [SW High Level Design](docs/nnfw/project/2018_high_level_design.md)

## Getting started

- For the contribution, please refer to our [contribution guide](docs/HowToContribute.md).
- You can also find how-to documents [HERE](docs/nnfw/howto.md).

## Maintainers

- Sung-Jae Lee <<sj925.lee@samsung.com>>
- Chunseok Lee <<chunseok.lee@samsung.com>>

## Committers

- Hyeongseok Oh <<hseok82.oh@samsung.com>>
- Hanjoung Lee <<hanjoung.lee@samsung.com>>
- Sharan Allur <<sharan.allur@samsung.com>>

## Feature Request (NEW)

You can suggest development of nnfw's features that are not yet available.

The functions requested so far can be checked in the [popular feature request](https://github.sec.samsung.net/STAR/nnfw/issues?utf8=%E2%9C%93&q=is%3Aopen+is%3Aissue+label%3AFEATURE_REQUEST+sort%3Areactions-%2B1-desc) list.

- If the feature you want is on the list, :+1: to the body of the issue. The feature with the most
:+1: is placed at the top of the list. When adding new features, we will prioritize them with this reference.
Of course, it is good to add an additional comment which describes your request in detail.

- For features not listed, [create a new issue](https://github.sec.samsung.net/STAR/nnfw/issues/new).
Sooner or later, the maintainer will tag the `FEATURE_REQUEST` label and appear on the list.

We expect one of the most frequent feature requests would be the operator kernel implementation.
It is good to make a request, but it is better if you contribute by yourself. See the following guide,
[How to Implement Operator Kernel](docs/nnfw/howto/HowToAddNewOperation.md), for help.

We are looking forward to your participation.
Thank you in advance!

# nncc
Re-targetable neural network (NN) model compilation framework

## Goals
nncc, which stands for neural network compiler collection, aims to provide a general framework for
compiling a given NN model to an artifact that runs on various target devices such as CPU, GPU, or
NPU.

## Maintainers

- Sung-Jae Lee <<sj925.lee@samsung.com>>
- Jonghyun Park <<jh1302.park@samsung.com>>

## Committers

- Saehie Park <<saehie.park@samsung.com>>
- Hyeongseok Oh <<hseok82.oh@samsung.com>>
- Efimov Alexander <<a.efimov@samsung.com>>

## How to Contact

- Please post questions, issues, or suggestions into [Issues](https://github.sec.samsung.net/STAR/nnfw/issues).

----

## Notice

### 22/07/2019

Congratulations! On July 22nd, 2019, _nnfw_ repo and
[_nncc_](https://github.sec.samsung.net/STAR/nncc) repo are finally integrated into single one. Now
all activities related to the development of _nnas(Neural Network Acceleration Solution)_ will
proceed in this integrated _nnfw_ repo. The old _nncc_ repo will only be maintained for follow up on
remaining issues and for preserving development history. The following notice will remain in place
until the update of documents in integrated repo is complete.

### 02/05/2019

~~We are currently working on [_nncc_](https://github.sec.samsung.net/STAR/nncc) as a sibling project.
In our plan, the two projects will soon be integrated into one, and focusing on their roles as
front-end(_nncc_) and back-end(_nnfw_), respectively. It will accompany the physical combination of
the github repo.~~ You can find the latest roadmap of the integrated project
[here](https://github.sec.samsung.net/orgs/STAR/projects/1).
