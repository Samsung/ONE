add_executable(tflchef Driver.cpp)
target_link_libraries(tflchef tflchef_core)
target_link_libraries(tflchef safemain)

install(TARGETS tflchef DESTINATION bin)
