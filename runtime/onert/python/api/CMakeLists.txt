# CMakeLists.txt

# install pybind11
nnas_find_package(Pybind11 REQUIRED)
if(NOT Pybind11_FOUND)
  message(STATUS "Build onert/python: FAILED (Pybind11 is missing)")
  return()
endif()

# Add the nnfw api and Pybind11 header

# Add the pybind tools include
list(APPEND CMAKE_MODULE_PATH "${Pybind11_INCLUDE_DIRS}/../../share/cmake/pybind11/")
include(${Pybind11_INCLUDE_DIRS}/../../share/cmake/pybind11/pybind11Tools.cmake)

# Add the Python module
pybind11_add_module(nnfw_api_pybind src/nnfw_api_pybind.cc)

target_include_directories(nnfw_api_pybind PRIVATE include ../../api/include)
target_include_directories(nnfw_api_pybind PRIVATE ${Pybind11_INCLUDE_DIRS})
target_link_libraries(nnfw_api_pybind PRIVATE nnfw-dev)

# Install the Python module
install(TARGETS nnfw_api_pybind DESTINATION lib)
