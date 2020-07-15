nnas_include(TargetRequire)

unset(REQUIRED_TARGETS)
list(APPEND REQUIRED_TARGETS mio_tflite)
list(APPEND REQUIRED_TARGETS mio_circle)
TargetRequire_Return(${REQUIRED_TARGETS})

set(DRIVER "driver/Driver.cpp")
file(GLOB_RECURSE SOURCES "src/*.cpp")
add_executable(tflite2circle ${DRIVER} ${SOURCES})
target_include_directories(tflite2circle PRIVATE include)
target_include_directories(tflite2circle PRIVATE src)
target_link_libraries(tflite2circle arser)
target_link_libraries(tflite2circle safemain)
target_link_libraries(tflite2circle mio_tflite)
target_link_libraries(tflite2circle mio_circle)

install(TARGETS tflite2circle DESTINATION bin)
