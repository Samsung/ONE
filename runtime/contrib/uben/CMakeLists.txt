if(NOT BUILD_UBEN)
  return()
endif(NOT BUILD_UBEN)

nnfw_find_package(ARMCompute QUIET)
nnfw_find_package(Nonius QUIET)

if(NOT ARMCompute_FOUND)
  return()
endif(NOT ARMCompute_FOUND)

if(NOT Nonius_FOUND)
  return()
endif(NOT Nonius_FOUND)

# 3x3 Convolution with unit stride
add_executable(uben_conv_3x3 Convolution.cpp)
target_compile_definitions(uben_conv_3x3 PRIVATE KER_H=3 KER_W=3 STRIDE_H=1 STRIDE_W=1)
target_compile_definitions(uben_conv_3x3 PRIVATE CL_DIRECT_CONVOLUTION=1)
target_compile_definitions(uben_conv_3x3 PRIVATE CL_GEMM_CONVOLUTION=1)
target_compile_definitions(uben_conv_3x3 PRIVATE CL_WINOGRAD_CONVOLUTION=1)
target_link_libraries(uben_conv_3x3 PRIVATE nonius)
target_link_libraries(uben_conv_3x3 PRIVATE arm_compute)
target_link_libraries(uben_conv_3x3 PRIVATE pthread)

add_executable(uben_softmax Softmax.cpp)
target_link_libraries(uben_softmax PRIVATE nonius)
target_link_libraries(uben_softmax PRIVATE nnfw_lib_cker)
target_link_libraries(uben_softmax PRIVATE pthread)
