if (NOT BUILD_RUNTIME_NNFW_API_TEST)
  return()
endif(NOT BUILD_RUNTIME_NNFW_API_TEST)

if (NOT BUILD_ONERT)
  message(STATUS "Skip build NNFW API test: no runtime build")
  return()
endif(NOT BUILD_ONERT)

nnfw_find_package(GTest)

set(RUNTIME_NNFW_API_TEST nnfw_api_gtest)
file(GLOB_RECURSE RUNTIME_NNFW_API_TEST_SRC "src/*.cc" "src/*.cpp")

add_executable(${RUNTIME_NNFW_API_TEST} ${RUNTIME_NNFW_API_TEST_SRC})

nnfw_find_package(ARMCompute QUIET)
if(ARMCompute_FOUND)
  target_compile_definitions(${RUNTIME_NNFW_API_TEST} PRIVATE TEST_ACL_BACKEND)
endif(ARMCompute_FOUND)

nnfw_find_package(Xnnpack QUIET)
if(Xnnpack_FOUND)
  target_compile_definitions(${RUNTIME_NNFW_API_TEST} PRIVATE TEST_XNNPACK_BACKEND)
endif(Xnnpack_FOUND)

set(RUNTIME_NNFW_API_TEST_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                  ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(${RUNTIME_NNFW_API_TEST} PRIVATE ${RUNTIME_NNFW_API_TEST_INCLUDE})

target_link_libraries(${RUNTIME_NNFW_API_TEST} nnfw-dev)
target_link_libraries(${RUNTIME_NNFW_API_TEST} gtest gmock)
target_link_libraries(${RUNTIME_NNFW_API_TEST} ${LIB_PTHREAD} dl)
target_link_libraries(${RUNTIME_NNFW_API_TEST} circle_schema)

install(TARGETS ${RUNTIME_NNFW_API_TEST} DESTINATION unittest_standalone)

# Pre-Install nnpackage test model (add)
set(NNPACKAGE_MODEL_DIR ${NNAS_PROJECT_SOURCE_DIR}/nnpackage/examples/one_op_in_tflite)
set(NNPACKAGE_INSTALL_TARGET unittest_standalone/nnfw_api_gtest_models)

set(NNPACKAGE_MODEL ${NNPACKAGE_MODEL_DIR}/add.tflite)
set(NNPACKAGE_METADATA ${NNPACKAGE_MODEL_DIR}/metadata/MANIFEST)

install(FILES ${NNPACKAGE_MODEL} DESTINATION ${NNPACKAGE_INSTALL_TARGET}/add/add)
install(FILES ${NNPACKAGE_METADATA} DESTINATION ${NNPACKAGE_INSTALL_TARGET}/add/add/metadata)

# Pre-Install nnpackage test model (add_no_manifest)
install(FILES ${NNPACKAGE_MODEL} DESTINATION ${NNPACKAGE_INSTALL_TARGET}/add_no_manifest/add_no_manifest)
