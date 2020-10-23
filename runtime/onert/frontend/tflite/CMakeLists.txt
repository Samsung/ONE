if(NOT BUILD_TFLITE_LOADER)
  return()
endif(NOT BUILD_TFLITE_LOADER)

set(TFLITE_LOADER_SOURCES src/tflite_loader.cc)

add_library(tflite_loader SHARED ${TFLITE_LOADER_SOURCES})

target_include_directories(tflite_loader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(tflite_loader PRIVATE onert_core)
target_link_libraries(tflite_loader PRIVATE base_loader nnfw_common nnfw_coverage)

if(CMAKE_BUILD_TYPE_LC STREQUAL "release")
  add_custom_command(TARGET tflite_loader POST_BUILD
                     COMMAND ${CMAKE_STRIP} "--strip-unneeded" $<TARGET_FILE_NAME:tflite_loader>)
endif()

install(TARGETS tflite_loader DESTINATION lib)
