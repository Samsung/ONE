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

set(RUNTIME_NNFW_API_TEST_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                  ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(${RUNTIME_NNFW_API_TEST} PRIVATE ${RUNTIME_NNFW_API_TEST_INCLUDE})

target_link_libraries(${RUNTIME_NNFW_API_TEST} nnfw-dev)
target_link_libraries(${RUNTIME_NNFW_API_TEST} gtest gmock)
target_link_libraries(${RUNTIME_NNFW_API_TEST} ${LIB_PTHREAD} dl)

install(TARGETS ${RUNTIME_NNFW_API_TEST} DESTINATION unittest)

# Install sample models for test
set(RUNTIME_NNFW_API_TEST_MODEL_DIR "${RUNTIME_NNFW_API_TEST}_models")
file(MAKE_DIRECTORY ${RUNTIME_NNFW_API_TEST_MODEL_DIR})
install(DIRECTORY ../../nnpackage/examples/one_op_in_tflite
        DESTINATION unittest/${RUNTIME_NNFW_API_TEST_MODEL_DIR}
        OPTIONAL)
