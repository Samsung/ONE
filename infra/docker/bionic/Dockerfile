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

# Build tool
RUN apt-get update && apt-get -qqy install build-essential cmake scons git g++-arm-linux-gnueabihf g++-aarch64-linux-gnu

# Install extra dependencies (Caffe, nnkit)
RUN apt-get update && apt-get -qqy install libboost-all-dev libgflags-dev libgoogle-glog-dev libatlas-base-dev libhdf5-dev

# Install protocol buffer
RUN apt-get update && apt-get -qqy install libprotobuf-dev protobuf-compiler

# Additonal tools
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -qqy install doxygen graphviz wget zip unzip clang-format-3.9 python3 python3-pip python3-venv hdf5-tools pylint
RUN pip3 install --upgrade pip
RUN pip3 install yapf==0.22.0 numpy

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

# Install lcov 1.13-4 for gcc-8 support (installed lcov 1.13-3 can't support gcc-8)
RUN wget http://launchpadlibrarian.net/370213541/lcov_1.13-4_all.deb
RUN dpkg -i lcov_1.13-4_all.deb

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
RUN wget http://download.tizen.org/sdk/tizenstudio/official/binary/sdb_3.1.4_ubuntu-64.zip -O sdb.zip
RUN unzip -d tmp sdb.zip && rm sdb.zip
RUN cp tmp/data/tools/sdb /usr/bin/. && rm -rf tmp

# Install java
RUN apt-get install -y --no-install-recommends openjdk-8-jdk

# download and install Gradle
# https://services.gradle.org/distributions/
ARG GRADLE_VERSION=6.4.1
ARG GRADLE_DIST=bin
RUN cd /opt && \
    wget -q https://services.gradle.org/distributions/gradle-${GRADLE_VERSION}-${GRADLE_DIST}.zip && \
    unzip gradle*.zip && \
    ls -d */ | sed 's/\/*$//g' | xargs -I{} mv {} gradle && \
    rm gradle*.zip

# download and install Android SDK
# https://developer.android.com/studio#command-tools
ARG ANDROID_SDK_VERSION=6514223
ENV ANDROID_SDK_ROOT /opt/android-sdk
RUN mkdir -p ${ANDROID_SDK_ROOT}/cmdline-tools && \
    wget -q https://dl.google.com/android/repository/commandlinetools-linux-${ANDROID_SDK_VERSION}_latest.zip && \
    unzip *tools*linux*.zip -d ${ANDROID_SDK_ROOT}/cmdline-tools && \
    rm *tools*linux*.zip

# accept the license agreements of the SDK components
RUN mkdir -p ${ANDROID_SDK_ROOT}/licenses
RUN echo 24333f8a63b6825ea9c5514f83c2829b004d1fee > ${ANDROID_SDK_ROOT}/licenses/android-sdk-license
RUN echo d56f5187479451eabf01fb78af6dfcb131a6481e >> ${ANDROID_SDK_ROOT}/licenses/android-sdk-license

# Env variable for gradle build
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64
ENV GRADLE_HOME /opt/gradle
ENV PATH ${PATH}:${GRADLE_HOME}/bin:${ANDROID_SDK_ROOT}/cmdline-tools/tools/bin:${ANDROID_SDK_ROOT}/platform-tools
ENV ANDROID_HOME ${ANDROID_SDK_ROOT}

# Install NDK
RUN sdkmanager --install "ndk;20.0.5594570"
RUN sdkmanager "platform-tools"

# Env for ko encoding build
ENV LC_ALL "C.UTF-8"

# setup adb server
EXPOSE 5037

# Clean archives (to reduce image size)
RUN apt-get clean -y
