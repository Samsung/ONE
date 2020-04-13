set(OPTIMIZATIONS_SRC
    CombineTransposes.cpp
    ConstantFoldTranspose.cpp
    FuseArithmeticOps.cpp
    DeadCodeElimination.cpp
    SinkRelu.cpp
    SinkTranspose.cpp
    OptimizationUtils.cpp)
nnc_add_library(nnc_optimizations SHARED ${OPTIMIZATIONS_SRC})
target_link_libraries(nnc_optimizations PRIVATE mir)

# install optimizations library
nnc_install_library(nnc_optimizations)
