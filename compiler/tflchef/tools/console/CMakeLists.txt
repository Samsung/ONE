add_executable(tflchef Driver.cpp)
target_link_libraries(tflchef tflchef_core)
target_link_libraries(tflchef safemain)

install(TARGETS tflchef DESTINATION bin)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

GTest_AddTest(tflchef_test Driver.test.cpp Driver.cpp)
target_link_libraries(tflchef_test tflchef_core)
