file(GLOB_RECURSE SOURCES "src/*.cpp")

# making gen-tf-input
add_executable(gen-tf-input ${SOURCES})
