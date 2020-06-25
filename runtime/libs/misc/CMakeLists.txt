# Library `nnfw_lib_misc`
file(GLOB_RECURSE NNFW_UTILITY_SRCS "src/*.cpp")

add_library(nnfw_lib_misc STATIC ${NNFW_UTILITY_SRCS})
target_include_directories(nnfw_lib_misc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(nnfw_lib_misc PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(nnfw_lib_misc PRIVATE nnfw_common)
target_link_libraries(nnfw_lib_misc PRIVATE nnfw_coverage)

# TODO Remove header dependency
install(DIRECTORY "include/misc"
        DESTINATION "include/onert" # FIXME This is only for onert developers
        FILES_MATCHING PATTERN "*.h"
        )

add_executable(nnfw_tensor_index_iterator "examples/tensor_index_iterator.cpp")
target_link_libraries(nnfw_tensor_index_iterator nnfw_lib_misc)
