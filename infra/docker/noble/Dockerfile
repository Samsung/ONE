# Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

FROM ubuntu:noble

# Install tzdata without user input
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -qqy install tzdata

# Build tool
RUN apt-get update && apt-get -qqy install build-essential cmake scons git lcov g++-arm-linux-gnueabihf g++-aarch64-linux-gnu pkg-config ccache

# Debian build tool
RUN apt-get update && apt-get -qqy install fakeroot devscripts debhelper python3-all dh-python

# Install extra dependencies (Caffe, nnkit)
RUN apt-get update && apt-get -qqy install libboost-all-dev libgflags-dev libgoogle-glog-dev libatlas-base-dev libhdf5-dev

# Additonal tools
RUN apt-get update && \
    apt-get -qqy install doxygen graphviz wget zip unzip python3 python3-pip python3-venv python3-dev hdf5-tools curl
RUN python3 -m pip install numpy flatbuffers --break-system-packages

# Setup ccache max size: 1.5GB
RUN ccache -M 1536MB

# Setup user ubuntu: give superuser permissions without password
RUN apt-get update && apt-get -qqy install sudo
RUN usermod -aG sudo ubuntu
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Clean archives (to reduce image size)
RUN apt-get clean -y

# Set user to ubuntu(1000)
USER ubuntu
WORKDIR /home/ubuntu
RUN python3 -m venv ~/.local
ENV PATH=~/.local/bin:$PATH
