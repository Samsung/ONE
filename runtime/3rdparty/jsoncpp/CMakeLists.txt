file(GLOB_RECURSE SRCS "*.cpp")

add_library(jsoncpp STATIC ${SRCS})
target_link_libraries(jsoncpp PRIVATE nnfw_common)
set_property(TARGET jsoncpp PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET jsoncpp APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
