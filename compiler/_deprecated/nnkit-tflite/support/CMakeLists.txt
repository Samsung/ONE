file(GLOB_RECURSE SOURCES "src/*.cpp")

# TODO Rename nnkit_support_tflite-1.13.1 as nnkit_tflite_support-1.13.1
add_library(nnkit_support_tflite-1.13.1 STATIC ${SOURCES})
set_target_properties(nnkit_support_tflite-1.13.1 PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(nnkit_support_tflite-1.13.1 PUBLIC include)
target_link_libraries(nnkit_support_tflite-1.13.1 nnkit_intf_backend)
target_link_libraries(nnkit_support_tflite-1.13.1 tensorflowlite-1.13.1)

add_library(nnkit_support_tflite ALIAS nnkit_support_tflite-1.13.1)
