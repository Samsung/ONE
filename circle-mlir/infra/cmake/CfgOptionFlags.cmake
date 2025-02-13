#
# configuration options
#
option(ONNX2CIRCLE_TEST_MODELS_SINGLE "Run onnx2cirle-models test with one at a time" OFF)

option(ENABLE_TEST "Enable module unit test and integration test" ON)
option(ENABLE_COVERAGE "Enable build for coverage test" OFF)

if(${ENABLE_COVERAGE} AND NOT ${ENABLE_TEST})
  message(FATAL_ERROR "Test should be enabled to measure test coverage")
endif()
