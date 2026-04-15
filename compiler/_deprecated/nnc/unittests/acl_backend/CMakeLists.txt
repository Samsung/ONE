set(ACL_CPP_BACKEND_UTEST_SOURCES DOMToText.cpp MIRToDOM.cpp)

file(GLOB_RECURSE ACL_IN_SOURCES "${NNC_ACL_BACKEND_DIR}/*.in")
nnc_make_generated_sources("${ACL_IN_SOURCES}" ${CMAKE_CURRENT_BINARY_DIR} ACL_GENERATED_SOURCES)

nnc_add_unit_test(nnc_acl_cpp_backend_test ${ACL_CPP_BACKEND_UTEST_SOURCES} ${ACL_GENERATED_SOURCES})

optional_target_link_libraries(nnc_acl_cpp_backend_test mir acl_soft_backend_cpp)
target_include_directories(nnc_acl_cpp_backend_test PRIVATE ${NNC_ACL_BACKEND_DIR} ${CMAKE_CURRENT_BINARY_DIR})
