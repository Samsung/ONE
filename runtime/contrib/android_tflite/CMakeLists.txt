if(NOT BUILD_ANDROID_TFLITE)
  return()
endif(NOT BUILD_ANDROID_TFLITE)

nnfw_find_package(TensorFlowLite EXACT 1.13.1 REQUIRED)

if(NOT DEFINED NDK_DIR)
  message(FATAL_ERROR "NDK_DIR should be specified via environment variable")
endif()
message(STATUS "Found NDK: ${NDK_DIR}")

#
# Tensorflow Lite JNI library
#
set(TENSORFLOW_LITE_BASE "${TensorFlowSource_DIR}/tensorflow/lite")
set(TFLITE_JNI_BASE ${TENSORFLOW_LITE_BASE}/java/src/main/native)
set(TFLITE_JNI_SRCS ${TFLITE_JNI_BASE}/exception_jni.cc
                    ${TFLITE_JNI_BASE}/nativeinterpreterwrapper_jni.cc
                    ${TFLITE_JNI_BASE}/tensor_jni.cc
                    ${TFLITE_JNI_BASE}/tensorflow_lite_jni.cc
                    ${CMAKE_CURRENT_SOURCE_DIR}/builtin_ops_jni.cc # Use nnfw's OpResolver
                    )
set(TFLITE_JNI_INCLUDES ${TENSORFLOW_LITE_BASE}/java/src/native)

# TODO use tensorflow-lite static library instead of compiling all the sources again
add_library(tensorflowlite_jni SHARED ${TFLITE_JNI_SRCS} ${TFLITE_SRCS})
target_include_directories(tensorflowlite_jni PUBLIC ${TFLITE_JNI_INCLUDES} ${TFLITE_INCLUDES})
target_link_libraries(tensorflowlite_jni eigen ${LIB_PTHREAD} dl)
target_link_libraries(tensorflowlite_jni log)
target_link_libraries(tensorflowlite_jni nnfw_lib_tflite)
install(TARGETS tensorflowlite_jni DESTINATION lib)
