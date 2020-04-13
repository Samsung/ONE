file(GLOB_RECURSE TESTS "*.cpp")

add_definitions(-DTEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test_data/")
nnc_add_unit_test(nnc_caffe_frontend_test ${TESTS})
optional_target_link_libraries(nnc_caffe_frontend_test mir_caffe_importer mir)
