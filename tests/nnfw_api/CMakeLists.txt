if (NOT BUILD_RUNTIME_NNFW_API_TEST)
  return()
endif(NOT BUILD_RUNTIME_NNFW_API_TEST)

if (NOT BUILD_ONERT)
  message(STATUS "Skip build NNFW API test: no runtime build")
  return()
endif(NOT BUILD_ONERT)

nnfw_find_package(GTest)

set(RUNTIME_NNFW_API_TEST nnfw_api_gtest)
file(GLOB_RECURSE RUNTIME_NNFW_API_TEST_LIB "lib/*.cc" "lib/*.cpp")
file(GLOB_RECURSE RUNTIME_NNFW_API_TEST_SRC "main.cc" "src/*.test.cc" "src/*.test.cpp")

add_executable(${RUNTIME_NNFW_API_TEST} ${RUNTIME_NNFW_API_TEST_LIB} ${RUNTIME_NNFW_API_TEST_SRC})

nnfw_find_package(ARMCompute QUIET)
if(ARMCompute_FOUND)
  target_compile_definitions(${RUNTIME_NNFW_API_TEST} PRIVATE TEST_ACL_BACKEND)
endif(ARMCompute_FOUND)

nnfw_find_package(Xnnpack QUIET)
if(Xnnpack_FOUND)
  target_compile_definitions(${RUNTIME_NNFW_API_TEST} PRIVATE TEST_XNNPACK_BACKEND)
endif(Xnnpack_FOUND)

nnas_find_package(Opencl_Headers QUIET)
if(Opencl_Headers_FOUND)
  target_compile_definitions(${RUNTIME_NNFW_API_TEST} PRIVATE TEST_GPU_CL_BACKEND)
endif(Opencl_Headers_FOUND)

# TODO: When circle_traininfo_generated.h is exported, fix this
set(CIRCLE_PLUS_HEADER_DIR ${NNAS_PROJECT_SOURCE_DIR}/runtime/onert/core/src/loader)
set(RUNTIME_NNFW_API_TEST_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/lib ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(${RUNTIME_NNFW_API_TEST} PRIVATE ${RUNTIME_NNFW_API_TEST_INCLUDE} ${CIRCLE_PLUS_HEADER_DIR})

target_link_libraries(${RUNTIME_NNFW_API_TEST} nnfw-dev jsoncpp)
target_link_libraries(${RUNTIME_NNFW_API_TEST} gtest gmock)
target_link_libraries(${RUNTIME_NNFW_API_TEST} ${LIB_PTHREAD} dl)
target_link_libraries(${RUNTIME_NNFW_API_TEST} circle_schema)

install(TARGETS ${RUNTIME_NNFW_API_TEST} DESTINATION unittest)

# Install nnpackage test model (add)
set(NNPACKAGE_MODEL_DIR ${NNAS_PROJECT_SOURCE_DIR}/nnpackage/examples/v1.0.0/add)
set(NNPACKAGE_INSTALL_TARGET unittest/nnfw_api_gtest_models)

install(DIRECTORY ${NNPACKAGE_MODEL_DIR} DESTINATION ${NNPACKAGE_INSTALL_TARGET}/add)

# Install nnpackage test model (add_no_manifest)
set(NNPACKAGE_MODEL ${NNPACKAGE_MODEL_DIR}/add.tflite)
install(FILES ${NNPACKAGE_MODEL} DESTINATION ${NNPACKAGE_INSTALL_TARGET}/add_no_manifest/add_no_manifest)

# Install nnpackage test model (add_invalid_manifest)
set(NNPACKAGE_MODEL_DIR ${NNAS_PROJECT_SOURCE_DIR}/nnpackage/examples/v1.0.0/add_invalid_manifest)
install(DIRECTORY ${NNPACKAGE_MODEL_DIR} DESTINATION ${NNPACKAGE_INSTALL_TARGET}/add_invalid_manifest)

# Install nnpackage test model (if)
set(NNPACKAGE_MODEL_DIR ${NNAS_PROJECT_SOURCE_DIR}/nnpackage/examples/v1.0.0/if_dynamic)
install(DIRECTORY ${NNPACKAGE_MODEL_DIR} DESTINATION ${NNPACKAGE_INSTALL_TARGET}/if_dynamic)

# Install nnpackage test model (while)
set(NNPACKAGE_MODEL_DIR ${NNAS_PROJECT_SOURCE_DIR}/nnpackage/examples/v1.0.0/while_dynamic)
install(DIRECTORY ${NNPACKAGE_MODEL_DIR} DESTINATION ${NNPACKAGE_INSTALL_TARGET}/while_dynamic)

# Install nnpackage test model (mobilenet)
set(NNPACKAGE_MODEL_DIR ${NNAS_PROJECT_SOURCE_DIR}/runtime/contrib/TFLiteSharp/TFLiteTestApp/res/)
install(DIRECTORY ${NNPACKAGE_MODEL_DIR} DESTINATION ${NNPACKAGE_INSTALL_TARGET}/mobilenet_v1_1.0_224)

