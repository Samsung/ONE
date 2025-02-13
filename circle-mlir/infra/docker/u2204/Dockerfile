FROM ubuntu:22.04 AS build

ENV DEBIAN_FRONTEND=noninteractive

# Install 'add-apt-repository'
RUN apt-get update \
 && apt-get -qqy install apt-utils software-properties-common

# Build tool
RUN apt-get update \
 && apt-get -qqy install build-essential cmake git fakeroot autoconf automake libtool unzip wget \
                         devscripts debmake debhelper lcov

# Install clang-format
RUN apt-get update \
 && apt-get install -qqy gnupg2 \
 && wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - \
 && add-apt-repository "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-16 main" \
 && apt-get update \
 && apt-get install -qqy clang-format-16

# additonal tools
RUN apt-get update \
 && apt-get -qqy install tree tmux

# Python
RUN apt-get update \
 && apt-get -qqy install python3 python3-pip python3-venv python3-dev python3-all dh-python
RUN python3 -m pip install --upgrade pip setuptools
RUN python3 -m pip install yapf==0.43.0 numpy h5py==3.8.0 einops

# TODO upgrade
ARG VER_TORCH=2.1.0
ARG VER_ONNX=1.16.0
ARG VER_ONNXRUNTIME=1.18.0

RUN python3 -m pip install torch==${VER_TORCH}
RUN python3 -m pip install onnx==${VER_ONNX}
RUN python3 -m pip install onnxruntime==${VER_ONNXRUNTIME}

# Clean archives (to reduce image size)
RUN apt-get clean -y

#
# For CIRCLE-MLIR
#
ARG NPROC=4
ARG WORK_DIR=/workdir
ARG FLATBUFFERS_HASH=a078130c
ARG ABSEIL_CPP_HASH=21510581
ARG LLVM_PROJECT_HASH=91088978d712cd7b33610c59f69d87d5a39e3113
ARG ONNX_MLIR_HASH=40615b362fe474d6cce4fd9c31a7ec51acd8f88a

WORKDIR ${WORK_DIR}

# Generate experimental patch file for onnx-mlir

RUN set -x \
 && { \
echo 'diff --git a/src/Dialect/ONNX/ONNXOps.td.inc b/src/Dialect/ONNX/ONNXOps.td.inc'; \
echo 'index d75ee6b6..f63a2910 100644'; \
echo '--- a/src/Dialect/ONNX/ONNXOps.td.inc'; \
echo '+++ b/src/Dialect/ONNX/ONNXOps.td.inc'; \
echo '@@ -1779,9 +1779,9 @@ def ONNXDequantizeLinearOp:ONNX_Op<"DequantizeLinear",'; \
echo '   `zero-point` is usually not used in the case of float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz quantization,'; \
echo "   but the dequantization formula remains the same for consistency and 'x_scale' still determines the output type."; \
echo '   }];'; \
echo '-  let arguments = (ins AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[I32]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$x,'; \
echo '+  let arguments = (ins AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$x,'; \
echo '     AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[BF16]>]>:$x_scale,'; \
echo '-    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[I32]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, NoneType]>:$x_zero_point,'; \
echo '+    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, NoneType]>:$x_zero_point,'; \
echo '     DefaultValuedAttr<SI64Attr, "1">:$axis);'; \
echo '   let results = (outs AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[BF16]>]>:$y);'; \
echo '   let extraClassDeclaration = [{'; \
echo '@@ -5721,10 +5721,10 @@ def ONNXQuantizeLinearOp:ONNX_Op<"QuantizeLinear",'; \
echo '   }];'; \
echo '   let arguments = (ins AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[I32]>]>:$x,'; \
echo '     AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[I32]>]>:$y_scale,'; \
echo '-    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, NoneType]>:$y_zero_point,'; \
echo '+    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[I16]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, NoneType]>:$y_zero_point,'; \
echo '     DefaultValuedAttr<SI64Attr, "1">:$axis,'; \
echo '     DefaultValuedAttr<SI64Attr, "1">:$saturate);'; \
echo '-  let results = (outs AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$y);'; \
echo '+  let results = (outs AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[I16]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$y);'; \
echo '   let extraClassDeclaration = [{'; \
echo '     static int getNumberOfOperands() {'; \
echo '       return 3;'; \
echo ''; \
} > ${WORK_DIR}/onnx_quantdequant_i16.patch \
 && cat ${WORK_DIR}/onnx_quantdequant_i16.patch

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

# ONNX-MLIR build from source
# Patch to accept INT16 for QuantizeLinear/DequantizeLinear
RUN cd onnx-mlir \
 && patch -p1 < ${WORK_DIR}/onnx_quantdequant_i16.patch

RUN mkdir onnx-mlir-build && cd onnx-mlir-build \
 && cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${WORK_DIR} \
    -DMLIR_DIR=${WORK_DIR}/llvm-project-build/lib/cmake/mlir \
    -DPython3_ROOT_DIR=/usr/bin \
    -DCMAKE_JOB_POOL_COMPILE:STRING=compile -DCMAKE_JOB_POOL_LINK:STRING=link \
    -DCMAKE_JOB_POOLS:STRING='compile=4;link=1' \
    -DONNX_MLIR_BUILD_TESTS=OFF -DONNX_MLIR_ENABLE_MHLO=OFF -DLLVM_ENABLE_ASSERTIONS=ON \
    -DONNX_MLIR_DECOMP_ONNX_CONVTRANSPOSE=OFF \
    ../onnx-mlir

RUN cd onnx-mlir-build && cmake --build . && cmake --install .

# LLVM/MLIR clean up
RUN rm -rf llvm-project
RUN rm -rf llvm-project-build

# ONNX-MLIR clean up
RUN rm -rf onnx-mlir/.git \
 && rm -rf onnx-mlir/third_party/* \
 && rm -rf onnx-mlir-build/test \
 && rm -rf onnx-mlir-build/Release
RUN find ./onnx-mlir-build -name *.o -exec rm {} \;
RUN find ./onnx-mlir-build -name *.so -exec rm {} \;

# bin cleanup and strip: 2.7G -> 1.2G
RUN rm -f ./bin/onnx-mlir-opt \
 && rm -f ./bin/onnx-mlir-reduce
RUN cd ./bin && strip -s *
RUN cd ./lib && strip -s *.so.*git

# more cleanup
RUN rm -f /usr/local/lib/libprotoc.* \
 && rm -rf ./onnx-mlir-build/docs/* \
 && rm -rf ./onnx-mlir-build/third_party/* \
 && rm -f ./onnx-mlir-build/.ninja_* \
 && rm -f ./onnx-mlir-build/build.ninja

# copy from build
FROM ubuntu:22.04

COPY --from=build / /

# Add user account
ARG USER_ACCOUNT=circlemlir

RUN groupadd -g 1000 ${USER_ACCOUNT} \
 && useradd -r -u 1000 -g ${USER_ACCOUNT} -ms /bin/bash ${USER_ACCOUNT}
USER ${USER_ACCOUNT}
WORKDIR /home/${USER_ACCOUNT}
CMD ["bash"]
