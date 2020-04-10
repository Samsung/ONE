nnfw_find_package(Eigen REQUIRED)
nnfw_find_package(GEMMLowp REQUIRED)
nnfw_find_package(Ruy REQUIRED)

add_library(nnfw_lib_cker INTERFACE)
target_link_libraries(nnfw_lib_cker INTERFACE eigen)
target_link_libraries(nnfw_lib_cker INTERFACE gemmlowp)
target_link_libraries(nnfw_lib_cker INTERFACE ruy)
target_include_directories(nnfw_lib_cker INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
