set(MIR_TEST_SOURCES Operation.cpp Index.cpp ShapeInference.cpp ShapeRange.cpp TensorVariant.cpp NodeReplacer.cpp)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

GTest_AddTest(mir_test ${MIR_TEST_SOURCES})
target_link_libraries(mir_test mir)
