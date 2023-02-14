#
# Simple Android JNI execution test of ACL
#

if(NOT ANDROID)
  return()
endif(NOT ANDROID)

nnfw_find_package(ARMCompute REQUIRED)

link_directories(${CMAKE_INSTALL_PREFIX}/lib)

set(JNIACL_SRCS src/jniacl_main.cc src/io_accessor.cc)

add_library(jniacl_jni SHARED ${JNIACL_SRCS})
target_include_directories(jniacl_jni PUBLIC ${TFLITE_JNI_INCLUDES} src)
target_link_libraries(jniacl_jni arm_compute_graph log)
