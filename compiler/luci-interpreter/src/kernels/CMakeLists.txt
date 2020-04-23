nnas_find_package(TensorFlowSource EXACT 2.1.0 REQUIRED)

# gemmlowp
nnas_include(ExternalSourceTools)
nnas_include(OptionTools)

# Exact version TensorFlow 2.1.0 uses.
envoption(TENSORFLOW_2_1_0_GEMMLOWP_URL https://github.com/google/gemmlowp/archive/12fed0cd7cfcd9e169bf1925bc3a7a58725fdcc3.zip)
ExternalSource_Download(TENSORFLOW_2_1_0_GEMMLOWP ${TENSORFLOW_2_1_0_GEMMLOWP_URL})
set(GEMMLOWP_SOURCE_DIR "${TENSORFLOW_2_1_0_GEMMLOWP_SOURCE_DIR}")

# Exact version TensorFlow 2.1.0 uses.
envoption(TENSORFLOW_2_1_0_EIGEN_URL https://gitlab.com/libeigen/eigen/-/archive/4e696901f873a2347f76d931cf2f701e31e15d05/eigen-4e696901f873a2347f76d931cf2f701e31e15d05.tar.gz)
ExternalSource_Download(TENSORFLOW_2_1_0_EIGEN ${TENSORFLOW_2_1_0_EIGEN_URL})
set(EIGEN_SOURCE_DIR "${TENSORFLOW_2_1_0_EIGEN_SOURCE_DIR}")

set(SOURCES
    Add.h
    Add.cpp
    AveragePool2D.h
    AveragePool2D.cpp
    Concatenation.h
    Concatenation.cpp
    Conv2D.h
    Conv2D.cpp
    DepthwiseConv2D.h
    DepthwiseConv2D.cpp
    FullyConnected.h
    FullyConnected.cpp
    MaxPool2D.h
    MaxPool2D.cpp
    Mul.h
    Mul.cpp
    Reshape.h
    Reshape.cpp
    Softmax.h
    Softmax.h
    Softmax.cpp
    Utils.h
    Utils.cpp)

add_library(luci_interpreter_kernels STATIC ${SOURCES})
set_target_properties(luci_interpreter_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(luci_interpreter_kernels PUBLIC ${LUCI_INTERPRETER_SOURCE_DIR})
target_link_libraries(luci_interpreter_kernels PUBLIC luci_interpreter_core)
target_link_libraries(luci_interpreter_kernels PRIVATE nncc_common)

# tensorflow
target_include_directories(luci_interpreter_kernels PRIVATE "${TensorFlowSource_DIR}")
target_compile_options(luci_interpreter_kernels PRIVATE -Wno-unused-parameter)

# gemmlowp
target_include_directories(luci_interpreter_kernels PRIVATE "${GEMMLOWP_SOURCE_DIR}")
target_compile_options(luci_interpreter_kernels PRIVATE -Wno-sign-compare)

# eigen
target_include_directories(luci_interpreter_kernels PRIVATE "${EIGEN_SOURCE_DIR}")
