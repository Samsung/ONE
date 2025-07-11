if(NOT BUILD_TFLITE_CLASSIFY_APP)
  return()
endif(NOT BUILD_TFLITE_CLASSIFY_APP)

list(APPEND SOURCES "src/tflite_classify.cc")
list(APPEND SOURCES "src/ImageClassifier.cc")
list(APPEND SOURCES "src/InferenceInterface.cc")

## Required package
find_package(OpenCV REQUIRED)

# Without this line, this appliation couldn't search the opencv library that were already installed in ${ROOTFS_DIR}/usr/lib/arm-linux-gnueabihf directory
set(CMAKE_EXE_LINKER_FLAGS "-Wl,--as-needed -Wl,--rpath=${ROOTFS_DIR}/usr/lib/arm-linux-gnueabihf -Wl,--rpath=${ROOTFS_DIR}/lib/arm-linux-gnueabihf")

add_executable(tflite_classify ${SOURCES})
target_include_directories(tflite_classify PRIVATE src)
target_link_libraries(tflite_classify tensorflow-lite ${LIB_PTHREAD} dl nnfw_lib_tflite)
target_link_libraries(tflite_classify ${OpenCV_LIBRARIES})

install(TARGETS tflite_classify DESTINATION ${CMAKE_INSTALL_BINDIR})
