FROM ubuntu:20.04 AS build

ENV DEBIAN_FRONTEND=noninteractive

# Install 'add-apt-repository'
RUN apt-get update \
 && apt-get -qqy install apt-utils software-properties-common

# Build tool
RUN apt-get update \
 && apt-get -qqy install build-essential git fakeroot autoconf automake libtool unzip wget \
                         devscripts debmake debhelper lcov
RUN apt-get update \
 && apt-get -qqy install zlib1g-dev pkg-config libhdf5-dev

# additonal tools
RUN apt-get update \
 && apt-get -qqy install tree tmux sudo

# add ppa:deadsnakes to install python3.10
RUN add-apt-repository ppa:deadsnakes/ppa -y
# add ppa:circletools to install circle-interpreter
RUN add-apt-repository ppa:circletools/nightly -y

# Python 3.10 as default
RUN apt-get update \
 && apt-get -qqy install python3.10 python3.10-venv python3.10-dev dh-python
RUN python3.10 -m ensurepip --upgrade

# TODO upgrade
ARG VER_TORCH=2.4.1+cpu
ARG VER_ONNX=1.17.0
ARG VER_ONNXRUNTIME=1.18.0
ARG VER_NUMPY=1.24.4

# Clean archives (to reduce image size)
RUN apt-get clean -y

# CMake 3.22
RUN wget https://cmake.org/files/v3.22/cmake-3.22.6-linux-x86_64.sh -O ${WORK_DIR}/cmake-3.22.6-linux-x86_64.sh \
 && bash ${WORK_DIR}/cmake-3.22.6-linux-x86_64.sh --prefix=/usr/local --skip-license
RUN rm ${WORK_DIR}/cmake-3.22.6-linux-x86_64.sh

#
# For CIRCLE-MLIR
#
ARG NPROC=4
ARG WORK_DIR=/workdir
ARG FLATBUFFERS_HASH=a078130c
ARG ABSEIL_CPP_HASH=d9e4955c65cd4367dd6bf46f4ccb8cd3d100540b
ARG LLVM_PROJECT_HASH=b270525f730be6e7196667925f5a9bfa153262e9
ARG ONNX_MLIR_HASH=d050bf35a2f186e45b0a4037c0a1d0727c69e7e6

WORKDIR ${WORK_DIR}

# prepare python3.10 virtual-env
# h5py and einops are used in test
RUN python3.10 -m venv ${WORK_DIR}/venv
RUN ${WORK_DIR}/venv/bin/python3 -m pip install --upgrade pip setuptools
RUN ${WORK_DIR}/venv/bin/python3 -m pip install numpy==${VER_NUMPY} torch==${VER_TORCH} \
    -f https://download.pytorch.org/whl/torch
RUN ${WORK_DIR}/venv/bin/python3 -m pip install onnx==${VER_ONNX} onnxruntime==${VER_ONNXRUNTIME} \
    h5py==3.8.0 einops

# Download experimental patch file for onnx-mlir

RUN wget https://raw.githubusercontent.com/Samsung/ONE/refs/heads/master/circle-mlir/externals/onnx_mlir_0_5_0_0.diff \
    -O ${WORK_DIR}/onnx_mlir_0_5_0_0.diff

# FLATBUFFERS build from source
RUN git clone --recursive https://github.com/google/flatbuffers.git \
 && cd flatbuffers && git checkout ${FLATBUFFERS_HASH}

RUN mkdir flatbuffers-build && cd flatbuffers-build \
 && cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${WORK_DIR} ../flatbuffers \
 && make -j${NPROC} && make install

RUN rm -rf flatbuffers \
 && rm -rf flatbuffers-build

# ABSEIL-CPP build from source
RUN git clone --recursive https://github.com/abseil/abseil-cpp.git \
 && cd abseil-cpp && git checkout ${ABSEIL_CPP_HASH}

RUN mkdir abseil-cpp-build && cd abseil-cpp-build \
 && cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${WORK_DIR} \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON \
    ../abseil-cpp \
 && make -j${NPROC} && make install

RUN rm -rf abseil-cpp \
 && rm -rf abseil-cpp-build

# NINJA
RUN git clone -b v1.10.2 https://github.com/ninja-build/ninja.git \
 && cd ninja && mkdir -p build && cd build \
 && cmake .. \
 && make -j${NPROC} install

RUN rm -rf ninja

# bazel to build protobuf from source
ARG BAZEL_ARCH="x86_64"
ARG BAZEL_VERSION=5.2.0
RUN wget -q https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-linux-${BAZEL_ARCH} \
    -O /usr/bin/bazel \
 && chmod a+x /usr/bin/bazel

# protobuf build from source
RUN git clone -b v21.12 https://github.com/google/protobuf.git \
 && cd protobuf \
 && git submodule update --init --recursive

RUN cd protobuf \
 && ./autogen.sh \
 && ./configure --with-zlib \
 && make -j${NPROC} && make install && ldconfig

RUN rm -rf protobuf \
 && rm -f /usr/bin/bazel

# Download LLVM and ONNX-MLIR
# NOTE download two huge repos at once to fast fail for network problems
RUN git clone --recursive https://github.com/llvm/llvm-project.git \
 && cd llvm-project && git checkout ${LLVM_PROJECT_HASH}

RUN git clone https://github.com/onnx/onnx-mlir.git \
 && cd onnx-mlir \
 && git checkout ${ONNX_MLIR_HASH} \
 && git submodule update --init --recursive

# LLVM/MLIR build from source
RUN mkdir llvm-project-build && cd llvm-project-build \
 && cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${WORK_DIR} \
    -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD=host \
    -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_RTTI=ON \
    -DCMAKE_JOB_POOL_COMPILE:STRING=compile -DCMAKE_JOB_POOL_LINK:STRING=link \
    -DCMAKE_JOB_POOLS:STRING='compile=4;link=1' \
    ../llvm-project/llvm

RUN cd llvm-project-build && cmake --build . && cmake --install .

# LLVM/MLIR clean up
RUN rm -rf llvm-project
RUN rm -rf llvm-project-build

# ONNX-MLIR build from source
# Patch to accept INT16 for QuantizeLinear/DequantizeLinear and fix ONNX passes
RUN cd onnx-mlir \
 && patch -p1 < ${WORK_DIR}/onnx_mlir_0_5_0_0.diff

RUN mkdir onnx-mlir-build && cd onnx-mlir-build \
 && cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${WORK_DIR} \
    -DMLIR_DIR=${WORK_DIR}/llvm-project-build/lib/cmake/mlir \
    -DPython3_ROOT_DIR=${WORK_DIR}/venv/bin \
    -DCMAKE_JOB_POOL_COMPILE:STRING=compile -DCMAKE_JOB_POOL_LINK:STRING=link \
    -DCMAKE_JOB_POOLS:STRING='compile=4;link=1' \
    -DONNX_MLIR_BUILD_TESTS=OFF -DONNX_MLIR_ENABLE_STABLEHLO=OFF \
    -DONNX_MLIR_ENABLE_JAVA=OFF -DLLVM_ENABLE_ASSERTIONS=ON \
    ../onnx-mlir

RUN cd onnx-mlir-build && cmake --build . && cmake --install .

# ONNX-MLIR clean up
RUN rm -rf onnx-mlir/.git \
 && rm -rf onnx-mlir/third_party/* \
 && rm -rf onnx-mlir-build/test \
 && rm -rf onnx-mlir-build/Release
RUN find ./onnx-mlir-build -name *.o -exec rm {} \;
RUN find ./onnx-mlir-build -name *.so -exec rm {} \;

# bin cleanup and strip
RUN rm -f ./bin/onnx-mlir-opt \
 && rm -f ./bin/onnx-mlir-reduce
RUN cd ./bin && strip -s *
RUN cd ./lib && strip -s *.so.*git

# more cleanup
RUN rm -f /usr/local/lib/libprotoc.* \
 && rm -rf /usr/local/man \
 && rm -rf /usr/local/doc \
 && rm -rf ./onnx-mlir-build/docs/* \
 && rm -rf ./onnx-mlir-build/third_party/* \
 && rm -f ./onnx-mlir-build/.ninja_* \
 && rm -f ./onnx-mlir-build/build.ninja

# copy from build
FROM ubuntu:20.04

COPY --from=build / /

# Add user account
ARG USER_ACCOUNT=circlemlir
ARG USER_ID=1000
ARG GROUP_ID=${USER_ID}

RUN addgroup --gid ${GROUP_ID} ${USER_ACCOUNT} \
 && adduser --disabled-password --gecos '' --uid ${USER_ID} --gid ${GROUP_ID} ${USER_ACCOUNT}  \
 && usermod -aG sudo ${USER_ACCOUNT}
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN echo 'circlemlir ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN echo 'source /workdir/venv/bin/activate' >> /etc/bash.bashrc

USER ${USER_ACCOUNT}
WORKDIR /home/${USER_ACCOUNT}
CMD ["bash"]
