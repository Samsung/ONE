file(GLOB_RECURSE API_SRC "*.cc")

set(NEURUN_DEV nnfw-dev)
add_library(${NEURUN_DEV} SHARED ${API_SRC})

# Public headers to publish
# nnfw_debug.h is header for runtime developer, so it will not be installed
# But runtime developer can use nnfw_debug.h by linking nnfw-dev
set(NNFW_API_HEADERS include/nnfw.h include/nnfw_dev.h)

target_link_libraries(${NEURUN_DEV} PUBLIC nnfw-nnapi-header)
target_link_libraries(${NEURUN_DEV} PRIVATE neurun_core)
target_link_libraries(${NEURUN_DEV} PRIVATE jsoncpp tflite_loader circle_loader ${LIB_PTHREAD})
target_link_libraries(${NEURUN_DEV} PRIVATE nnfw_common)
target_link_libraries(${NEURUN_DEV} PRIVATE nnfw_coverage)
target_include_directories(${NEURUN_DEV} PUBLIC include)
set_target_properties(${NEURUN_DEV} PROPERTIES PUBLIC_HEADER "${NNFW_API_HEADERS}")

install(TARGETS ${NEURUN_DEV}
        LIBRARY DESTINATION lib
        PUBLIC_HEADER DESTINATION include/nnfw)
