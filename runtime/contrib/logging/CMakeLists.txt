if(NOT BUILD_LOGGING)
  return()
endif(NOT BUILD_LOGGING)

file(GLOB_RECURSE NNAPI_LOGGING_SRCS "src/*.cc")

add_library(neuralnetworks SHARED ${NNAPI_LOGGING_SRCS})
target_include_directories(neuralnetworks PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(neuralnetworks PUBLIC nnfw-nnapi-header)
