nnas_find_package(GEMMLowp REQUIRED)

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
target_link_libraries(luci_interpreter_kernels PRIVATE gemmlowp)
target_link_libraries(luci_interpreter_kernels PRIVATE nncc_common)
