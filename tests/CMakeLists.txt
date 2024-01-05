if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

# Use C++17 for test
# TODO Use C++17 for all runtime directories
set(CMAKE_CXX_STANDARD 17)

add_subdirectories()
