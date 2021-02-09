add_library(foder INTERFACE)
target_include_directories(foder INTERFACE include)
target_link_libraries(foder INTERFACE nncc_coverage)
