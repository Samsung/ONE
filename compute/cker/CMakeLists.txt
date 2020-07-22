nnfw_find_package(Eigen REQUIRED)
nnfw_find_package(Ruy REQUIRED)
nnas_find_package(TensorFlowGEMMLowpSource EXACT 2.3.0 REQUIRED)

add_library(nnfw_lib_cker INTERFACE)
target_link_libraries(nnfw_lib_cker INTERFACE eigen)
target_link_libraries(nnfw_lib_cker INTERFACE ruy)
target_link_libraries(nnfw_lib_cker INTERFACE ruy_instrumentation)
target_compile_definitions(nnfw_lib_cker INTERFACE USE_RUY_GEMV)
if(EXPERIMENTAL_RUY_FEATURE)
  target_compile_definitions(nnfw_lib_cker INTERFACE EXPERIMENTAL_RUY_FEATURE)
endif(EXPERIMENTAL_RUY_FEATURE)
if(PROFILE_RUY)
  target_link_libraries(nnfw_lib_cker INTERFACE ruy_profiler)
endif(PROFILE_RUY)

target_include_directories(nnfw_lib_cker INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(nnfw_lib_cker SYSTEM INTERFACE ${TensorFlowGEMMLowpSource_DIR})

# Workaround to avoid warning
# TODO Resolve warning
target_compile_options(nnfw_lib_cker INTERFACE -Wno-attributes)
