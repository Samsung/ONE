add_executable(tflchef-reverse Driver.cpp)
target_link_libraries(tflchef-reverse arser)
target_link_libraries(tflchef-reverse tflchef_tflite)
target_link_libraries(tflchef-reverse safemain)
target_link_libraries(tflchef-reverse foder)

install(TARGETS tflchef-reverse DESTINATION bin)
