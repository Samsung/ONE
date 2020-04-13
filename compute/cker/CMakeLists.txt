nnfw_find_package(Eigen REQUIRED)

add_library(nnfw_lib_cker INTERFACE)
target_link_libraries(nnfw_lib_cker INTERFACE eigen)
target_include_directories(nnfw_lib_cker INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
