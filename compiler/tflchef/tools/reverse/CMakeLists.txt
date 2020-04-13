add_executable(tflchef-reverse Driver.cpp)
target_link_libraries(tflchef-reverse tflchef_tflite)
target_link_libraries(tflchef-reverse safemain)
