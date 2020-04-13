if(NOT BUILD_LOGGING)
  return()
endif(NOT BUILD_LOGGING)

file(GLOB_RECURSE NNAPI_LOGGING_SRCS "src/*.cc")

nnas_find_package(Boost REQUIRED)

add_library(neuralnetworks SHARED ${NNAPI_LOGGING_SRCS})
target_include_directories(neuralnetworks PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(neuralnetworks PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(neuralnetworks PUBLIC nnfw-nnapi-header)
