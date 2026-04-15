file(GLOB_RECURSE DUMPER_SRC ./*.cpp ./*.h)
nnc_add_library(nnc_dumper SHARED ${DUMPER_SRC})
target_link_libraries(nnc_dumper PRIVATE mir)

# install dumper library
nnc_install_library(nnc_dumper)
