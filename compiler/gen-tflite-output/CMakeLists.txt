file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(gen-tflite-output ${SOURCES})
