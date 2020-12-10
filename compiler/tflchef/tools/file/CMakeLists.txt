add_executable(tflchef-file Driver.cpp)
target_link_libraries(tflchef-file arser)
target_link_libraries(tflchef-file tflchef_core)
target_link_libraries(tflchef-file safemain)

install(TARGETS tflchef-file DESTINATION bin)
