FROM ubuntu:18.04
RUN apt update -y && \
    apt clean && \
    apt install -y \
    build-essential \
    libpixman-1-dev \
    libssl-dev \
    gcc-arm-none-eabi \
    autoconf \
    git \
    cmake \
    python-pip \
    libboost-all-dev \
    automake \
    make \
    software-properties-common && \
    pip install ninja && \
    apt install -y python3.8 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    apt clean
WORKDIR /root/src
RUN git clone https://github.com/qemu/qemu && \
    cd qemu && \
    git checkout 621da7789083b80d6f1ff1c0fb499334007b4f51 && \
    ./configure && \
    make -j$(nproc)
RUN git clone https://github.com/Kitware/CMake.git && \
    cd CMake && \
    git checkout c3793b41573ee584e572ea546527b9350f46a05d && \
    apt remove -y --purge --auto-remove cmake && \
    ./bootstrap && make -j$(nproc) && make install
RUN git clone https://github.com/gwilymk/bin2c.git bin2c && \
    cd bin2c && \
    gcc -o bin2c bin2c.c
#RUN git clone https://github.com/Samsung/ONE.git one && \
#    cd one && \
#    mkdir build && \
#    cd build && \
#    cmake ../infra/nncc && \
#    make -j$(nproc) luci_interpreter_micro_arm
