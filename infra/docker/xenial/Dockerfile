FROM ubuntu:16.04

ARG UBUNTU_MIRROR

RUN if [ -n "$http_proxy" ] ; then echo "Acquire::http::proxy \"${http_proxy}\";" >> /etc/apt/apt.conf ; fi
RUN if [ -n "$https_proxy" ] ; then echo "Acquire::https::proxy \"${https_proxy}\";" >> /etc/apt/apt.conf ; fi
RUN if [ -n "$UBUNTU_MIRROR" ] ; then sed "s/archive.ubuntu.com/${UBUNTU_MIRROR}/g" -i /etc/apt/sources.list ; fi

# Install 'add-apt-repository'
RUN apt-get update && apt-get -qqy install software-properties-common

# Build tool
RUN apt-get update && apt-get -qqy install build-essential cmake scons git lcov

# Install extra dependencies (Caffe, nnkit)
RUN apt-get update && apt-get -qqy install libboost-all-dev libgflags-dev libgoogle-glog-dev libatlas-base-dev libhdf5-dev

# Install protocol buffer
RUN apt-get update && apt-get -qqy install libprotobuf-dev protobuf-compiler

# Additonal tools
RUN apt-get update && \
    apt-get -qqy install doxygen graphviz wget unzip clang-format-3.9 clang-format-8 python3 python3-pip python3-venv hdf5-tools pylint curl
RUN pip3 install --upgrade pip
RUN pip3 install yapf==0.22.0 numpy

# Install google test (source)
RUN apt-get update && apt-get -qqy install libgtest-dev

###
### NOTE: Don't add new package install using apt-get or pip below this line
###

# Install native build tool gcc version 6.x
RUN add-apt-repository ppa:ubuntu-toolchain-r/test && apt-get update && apt-get -qqy install gcc-6 g++-6
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 60 --slave /usr/bin/g++ g++ /usr/bin/g++-6 && update-alternatives --config gcc

# Install cross build tool gcc version 6.x
RUN wget https://releases.linaro.org/components/toolchain/binaries/6.3-2017.02/arm-linux-gnueabihf/gcc-linaro-6.3.1-2017.02-x86_64_arm-linux-gnueabihf.tar.xz -O gcc-hardfp.tar.xz -nv
RUN wget https://releases.linaro.org/components/toolchain/binaries/6.2-2016.11/arm-linux-gnueabi/gcc-linaro-6.2.1-2016.11-x86_64_arm-linux-gnueabi.tar.xz -O gcc-softfp.tar.xz -nv
RUN wget https://releases.linaro.org/components/toolchain/binaries/6.2-2016.11/aarch64-linux-gnu/gcc-linaro-6.2.1-2016.11-x86_64_aarch64-linux-gnu.tar.xz -O gcc-aarch64.tar.xz -nv
RUN tar -xf gcc-hardfp.tar.xz -C /opt/ && rm -rf gcc-hardfp.tar.xz
RUN tar -xf gcc-softfp.tar.xz -C /opt/ && rm -rf gcc-softfp.tar.xz
RUN tar -xf gcc-aarch64.tar.xz -C /opt/ && rm -rf gcc-aarch64.tar.xz
ENV PATH "/opt/gcc-linaro-6.2.1-2016.11-x86_64_arm-linux-gnueabi/bin:/opt/gcc-linaro-6.3.1-2017.02-x86_64_arm-linux-gnueabihf/bin:/opt/gcc-linaro-6.2.1-2016.11-x86_64_aarch64-linux-gnu/bin:$PATH"

###
### NOTE: Don't add build & install process using installed buildtool above this line
###

# Build and install google test static libraries
WORKDIR /root/gtest
RUN cmake /usr/src/gtest
RUN make
RUN mv *.a /usr/lib
WORKDIR /root
RUN rm -rf gtest

# Install gbs & sdb
RUN echo 'deb [trusted=yes] http://download.tizen.org/tools/latest-release/Ubuntu_16.04/ /' | cat >> /etc/apt/sources.list
RUN apt-get update && apt-get -qqy install gbs
RUN wget http://download.tizen.org/sdk/tizenstudio/official/binary/sdb_3.1.4_ubuntu-64.zip -O sdb.zip
RUN unzip -d tmp sdb.zip && rm sdb.zip
RUN cp tmp/data/tools/sdb /usr/bin/. && rm -rf tmp

# Clean archives (to reduce image size)
RUN apt-get clean -y
