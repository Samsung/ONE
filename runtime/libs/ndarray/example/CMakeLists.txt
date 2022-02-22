add_executable(example_no_array example_no_array.cpp)

add_executable(example_array example_array.cpp)
target_link_libraries(example_array PRIVATE ndarray)
