# Copyright 2016-2020 Jing Li
# Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM ubuntu:18.04

ARG UBUNTU_MIRROR

# Install 'add-apt-repository'
RUN apt-get update && apt-get -qqy install software-properties-common

# Git repo for latest version (github checkout@v2 action requires v2.18)
RUN add-apt-repository ppa:git-core/ppa -y

# Build tool
RUN apt-get update && apt-get -qqy install build-essential cmake scons git g++-arm-linux-gnueabihf g++-aarch64-linux-gnu

# ARM none eabi build tool
RUN apt-get update && apt-get -qqy install gcc-arm-none-eabi

# Debian build tool
RUN apt-get update && apt-get -qqy install fakeroot devscripts debhelper python3-all

# Install extra dependencies (Caffe, nnkit)
RUN apt-get update && apt-get -qqy install libboost-all-dev libgflags-dev libgoogle-glog-dev libatlas-base-dev libhdf5-dev

# Install protocol buffer
RUN apt-get update && apt-get -qqy install libprotobuf-dev protobuf-compiler

# Additonal tools
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get -qqy install doxygen graphviz wget zip unzip python3 python3-pip python3-venv python3-dev hdf5-tools pylint curl
RUN apt-get update && apt-get -qqy install python3.8 python3.8-venv python3.8-dev
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install yapf==0.22.0 numpy flatbuffers
RUN python3.8 -m pip install --upgrade pip
RUN python3.8 -m pip install numpy flatbuffers

# Skip clang-format-16 install
# Cannot install clang-formart-16

# Install google test (source)
RUN apt-get update && apt-get -qqy install libgtest-dev

# Install build tool gcc version 8.x and set alternative link (c++17 support)
RUN apt-get update && apt-get -qqy install g++-8 g++-8-arm-linux-gnueabihf g++-8-aarch64-linux-gnu
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 80 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-8 \
    --slave /usr/bin/gcov gcov /usr/bin/gcov-8
RUN update-alternatives --install /usr/bin/arm-linux-gnueabihf-gcc arm-linux-gnueabihf-gcc /usr/bin/arm-linux-gnueabihf-gcc-8 80 \
    --slave /usr/bin/arm-linux-gnueabihf-g++ arm-linux-gnueabihf-g++ /usr/bin/arm-linux-gnueabihf-g++-8 \
    --slave /usr/bin/arm-linux-gnueabihf-gcov arm-linux-gnueabihf-gcov /usr/bin/arm-linux-gnueabihf-gcov-8
RUN update-alternatives --install /usr/bin/aarch64-linux-gnu-gcc aarch64-linux-gnu-gcc /usr/bin/aarch64-linux-gnu-gcc-8 80 \
    --slave /usr/bin/aarch64-linux-gnu-g++ aarch64-linux-gnu-g++ /usr/bin/aarch64-linux-gnu-g++-8 \
    --slave /usr/bin/aarch64-linux-gnu-gcov aarch64-linux-gnu-gcov /usr/bin/aarch64-linux-gnu-gcov-8

# Install lcov 1.14-2 for gcc-8 support
#   Default version lcov 1.13-3 can't support gcc-8
#   lcov 1.13-4 with gcc-8 have bug: reports no coverage for class declaration
WORKDIR /root/lcov
RUN wget http://archive.ubuntu.com/ubuntu/pool/universe/l/lcov/lcov_1.14-2_all.deb
RUN apt-get update && apt-get -qqy install libperlio-gzip-perl libjson-perl
RUN dpkg -i lcov_1.14-2_all.deb
WORKDIR /root
RUN rm -rf /root/lcov

# Build and install google test static libraries
WORKDIR /root/gtest
RUN cmake /usr/src/gtest
RUN make
RUN mv *.a /usr/lib
WORKDIR /root
RUN rm -rf gtest

# Install gbs & sdb
RUN echo 'deb [trusted=yes] http://download.tizen.org/tools/latest-release/Ubuntu_18.04/ /' | cat >> /etc/apt/sources.list
RUN apt-get update && apt-get -qqy install gbs
RUN wget http://download.tizen.org/sdk/tizenstudio/official/binary/sdb_4.2.25_ubuntu-64.zip -O sdb.zip
RUN unzip -d tmp sdb.zip && rm sdb.zip
RUN cp tmp/data/tools/sdb /usr/bin/. && rm -rf tmp/*

# Env for ko encoding build
ENV LC_ALL "C.UTF-8"

# setup adb server
EXPOSE 5037

# Setup user to match host user, and give superuser permissions
ARG USER_ID=1000
ARG GROUP_ID=${USER_ID}
RUN addgroup --gid ${GROUP_ID} ubuntu && adduser --disabled-password --gecos '' --uid ${USER_ID} --gid ${GROUP_ID} ubuntu && usermod -aG sudo ubuntu
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Clean archives (to reduce image size)
RUN apt-get clean -y

# Set user to the one we just created
USER ${USER_ID}
