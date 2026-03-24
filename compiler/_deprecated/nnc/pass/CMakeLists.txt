set(PASS_MANAGER_SRC PassManager.cpp)

nnc_add_library(nnc_pass STATIC ${PASS_MANAGER_SRC})
target_link_libraries(nnc_pass PUBLIC mir)
set_target_properties(nnc_pass PROPERTIES LINKER_LANGUAGE CXX)
