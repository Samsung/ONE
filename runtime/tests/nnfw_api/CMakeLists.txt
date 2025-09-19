if (NOT BUILD_RUNTIME_NNFW_API_TEST)
  return()
endif(NOT BUILD_RUNTIME_NNFW_API_TEST)

nnfw_find_package(GTest)

# Shared test utilities library
set(RUNTIME_NNFW_API_TEST_LIB_SRC
  lib/CircleGen.cc
  lib/CirclePlusGen.cc
  lib/common.cc
  lib/NNPackages.cc
)

add_library(nnfw_api_test_utils STATIC ${RUNTIME_NNFW_API_TEST_LIB_SRC})
set_target_properties(nnfw_api_test_utils PROPERTIES POSITION_INDEPENDENT_CODE ON)

nnfw_find_package(ARMCompute QUIET)
if(ARMCompute_FOUND)
  target_compile_definitions(nnfw_api_test_utils PUBLIC TEST_ACL_BACKEND)
endif(ARMCompute_FOUND)

if(BUILD_XNNPACK_BACKEND)
  nnfw_find_package(Xnnpack QUIET)
  if(Xnnpack_FOUND)
    target_compile_definitions(nnfw_api_test_utils PUBLIC TEST_XNNPACK_BACKEND)
  endif(Xnnpack_FOUND)
endif(BUILD_XNNPACK_BACKEND)

nnfw_find_package(Opencl_Headers QUIET)
if(Opencl_Headers_FOUND)
  target_compile_definitions(nnfw_api_test_utils PUBLIC TEST_GPU_CL_BACKEND)
endif(Opencl_Headers_FOUND)

set(RUNTIME_NNFW_API_TEST_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/lib ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(nnfw_api_test_utils PUBLIC ${RUNTIME_NNFW_API_TEST_INCLUDE})

target_link_libraries(nnfw_api_test_utils nnfw-dev jsoncpp circle_schema nnfw_common)

# Function to create test group executables
function(create_nnfw_test_group GROUP_NAME GROUP_SOURCES)
  file(GLOB_RECURSE GROUP_TEST_SRC
    "main.cc"
    ${GROUP_SOURCES}
  )

  set(EXECUTABLE_NAME nnfw_api_gtest_one_op_${GROUP_NAME})

  add_executable(${EXECUTABLE_NAME} ${GROUP_TEST_SRC})

  # Link shared library and other dependencies
  target_link_libraries(${EXECUTABLE_NAME} nnfw_api_test_utils)
  target_link_libraries(${EXECUTABLE_NAME} gtest gmock)
  target_link_libraries(${EXECUTABLE_NAME} Threads::Threads dl)

  install(TARGETS ${EXECUTABLE_NAME} DESTINATION unittest)
endfunction()

# Create test group executables based on operator classification
# See OPERATOR_CLASSIFICATION.md for detailed grouping rationale

# MathOps - Basic mathematical, comparison, and bitwise operations
create_nnfw_test_group("MathOps"
  "src/GenModelTests/one_op_tests/MathOps/*.test.cc"
  "src/GenModelTests/one_op_tests/MathOps/*.test.cpp")

# NeuralNetworkOps - Core neural network layer operations
create_nnfw_test_group("NeuralNetworkOps"
  "src/GenModelTests/one_op_tests/NeuralNetworkOps/*.test.cc"
  "src/GenModelTests/one_op_tests/NeuralNetworkOps/*.test.cpp")

# ActivationNormOps - Activation functions and normalization layers
create_nnfw_test_group("ActivationNormOps"
  "src/GenModelTests/one_op_tests/ActivationNormOps/*.test.cc"
  "src/GenModelTests/one_op_tests/ActivationNormOps/*.test.cpp")

# TensorDataOps - Operations that transform tensor data structures, shapes, layouts, and manipulate tensor data
create_nnfw_test_group("TensorDataOps"
  "src/GenModelTests/one_op_tests/TensorDataOps/*.test.cc"
  "src/GenModelTests/one_op_tests/TensorDataOps/*.test.cpp")

# IndexingDataOps - Data indexing, lookup, and generation operations
create_nnfw_test_group("IndexingDataOps"
  "src/GenModelTests/one_op_tests/IndexingDataOps/*.test.cc"
  "src/GenModelTests/one_op_tests/IndexingDataOps/*.test.cpp")

# ReductionStatsOps - Reduction operations and statistical computations
create_nnfw_test_group("ReductionStatsOps"
  "src/GenModelTests/one_op_tests/ReductionStatsOps/*.test.cc"
  "src/GenModelTests/one_op_tests/ReductionStatsOps/*.test.cpp")

# ControlFlowOps - Control flow and conditional execution operations
create_nnfw_test_group("ControlFlowOps"
  "src/GenModelTests/one_op_tests/ControlFlowOps/*.test.cc"
  "src/GenModelTests/one_op_tests/ControlFlowOps/*.test.cpp")

# TypeConversionOps - Data type conversion and quantization operations
create_nnfw_test_group("TypeConversionOps"
  "src/GenModelTests/one_op_tests/TypeConversionOps/*.test.cc"
  "src/GenModelTests/one_op_tests/TypeConversionOps/*.test.cpp")

# AdvancedSpecialOps - Advanced and special-purpose operations
create_nnfw_test_group("AdvancedSpecialOps"
  "src/GenModelTests/one_op_tests/AdvancedSpecialOps/*.test.cc"
  "src/GenModelTests/one_op_tests/AdvancedSpecialOps/*.test.cpp")

file(GLOB_RECURSE ONE_OP_TRAINS_SRC "src/GenModelTests/one_op_trains/*.test.cc" "src/GenModelTests/one_op_trains/*.test.cpp")
add_executable(nnfw_api_gtest_one_op_trains main.cc ${ONE_OP_TRAINS_SRC})
target_link_libraries(nnfw_api_gtest_one_op_trains nnfw_api_test_utils gtest gmock Threads::Threads dl)
install(TARGETS nnfw_api_gtest_one_op_trains DESTINATION unittest)

file(GLOB_RECURSE NONTRAINABLE_OP_TRAINS_SRC "src/GenModelTests/nontrainable_op_trains/*.test.cc" "src/GenModelTests/nontrainable_op_trains/*.test.cpp")
add_executable(nnfw_api_gtest_nontrainable_op_trains main.cc ${NONTRAINABLE_OP_TRAINS_SRC})
target_link_libraries(nnfw_api_gtest_nontrainable_op_trains nnfw_api_test_utils gtest gmock Threads::Threads dl)
install(TARGETS nnfw_api_gtest_nontrainable_op_trains DESTINATION unittest)

file(GLOB_RECURSE NNPACKAGE_TESTS_SRC "src/NNPackageTests/*.test.cc" "src/NNPackageTests/*.test.cpp")
add_executable(nnfw_api_gtest_nnpackage main.cc ${NNPACKAGE_TESTS_SRC})
target_link_libraries(nnfw_api_gtest_nnpackage nnfw_api_test_utils gtest gmock Threads::Threads dl)
install(TARGETS nnfw_api_gtest_nnpackage DESTINATION unittest)

# Install nnpackage test model (add)
set(NNPACKAGE_MODEL_DIR ${ONE_PROJECT_SOURCE_DIR}/nnpackage/examples/v1.0.0/add)
set(NNPACKAGE_INSTALL_TARGET unittest/nnfw_api_gtest_models)

install(DIRECTORY ${NNPACKAGE_MODEL_DIR} DESTINATION ${NNPACKAGE_INSTALL_TARGET}/add)

# Install nnpackage test model (add_no_manifest)
set(NNPACKAGE_MODEL ${NNPACKAGE_MODEL_DIR}/add.tflite)
install(FILES ${NNPACKAGE_MODEL} DESTINATION ${NNPACKAGE_INSTALL_TARGET}/add_no_manifest/add_no_manifest)

# Install nnpackage test model (add_invalid_manifest)
set(NNPACKAGE_MODEL_DIR ${ONE_PROJECT_SOURCE_DIR}/nnpackage/examples/v1.0.0/add_invalid_manifest)
install(DIRECTORY ${NNPACKAGE_MODEL_DIR} DESTINATION ${NNPACKAGE_INSTALL_TARGET}/add_invalid_manifest)

# Install nnpackage test model (if)
set(NNPACKAGE_MODEL_DIR ${ONE_PROJECT_SOURCE_DIR}/nnpackage/examples/v1.0.0/if_dynamic)
install(DIRECTORY ${NNPACKAGE_MODEL_DIR} DESTINATION ${NNPACKAGE_INSTALL_TARGET}/if_dynamic)

# Install nnpackage test model (while)
set(NNPACKAGE_MODEL_DIR ${ONE_PROJECT_SOURCE_DIR}/nnpackage/examples/v1.0.0/while_dynamic)
install(DIRECTORY ${NNPACKAGE_MODEL_DIR} DESTINATION ${NNPACKAGE_INSTALL_TARGET}/while_dynamic)

# Install nnpackage test model (mobilenet)
set(NNPACKAGE_MODEL_DIR ${ONE_PROJECT_SOURCE_DIR}/runtime/contrib/TFLiteSharp/TFLiteTestApp/res/)
install(DIRECTORY ${NNPACKAGE_MODEL_DIR} DESTINATION ${NNPACKAGE_INSTALL_TARGET}/mobilenet_v1_1.0_224)

# Install nnpackage test model (conv2d)
set(NNPACKAGE_MODEL_DIR ${ONE_PROJECT_SOURCE_DIR}/runtime/tests/nnfw_api/res/autocompilation/)
install(DIRECTORY ${NNPACKAGE_MODEL_DIR} DESTINATION ${NNPACKAGE_INSTALL_TARGET}/conv2d)
