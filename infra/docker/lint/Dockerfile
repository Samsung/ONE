# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

FROM ubuntu:24.04

# Install 'add-apt-repository'
RUN apt-get update && apt-get -qqy install software-properties-common

# Install tzdata without user input
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get -qqy install tzdata

# Additonal tools
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get -qqy install python3 python3-pip
RUN python3 -m pip install yapf==0.40.2 pylint --break-system-packages

# Install clang-format
RUN apt-get update && apt-get install -qqy clang-format-16

# Setup user ubuntu: give superuser permissions without password
RUN apt-get update && apt-get -qqy install sudo
RUN usermod -aG sudo ubuntu
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Clean archives (to reduce image size)
RUN apt-get clean -y

# Set user to ubuntu(1000)
USER ubuntu
