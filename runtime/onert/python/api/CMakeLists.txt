# CMakeLists.txt

# refer https://github.com/Samsung/ONE/issues/11368

# Set root directory of Python
set(PYTHON_EXECUTABLE ${ROOTFS_ARM}/usr/bin/python3.8)

# Disable pybind11 python search mechanism
set(PYTHONLIBS_FOUND TRUE CACHE INTERNAL "")

# Settings before running find_package
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY "Set to find packages only in cross root path by force")
find_package(Python COMPONENTS Interpreter Development)

# Tell pybind11 where the target python installation is
set(PYTHON_INCLUDE_DIRS ${Python_INCLUDE_DIRS} CACHE INTERNAL "Cross python include path")
set(PYTHON_LIBRARIES ${Python_LIBRARIES} CACHE INTERNAL "Cross python include path")
set(PYTHON_MODULE_PREFIX "lib" CACHE INTERNAL "Cross python lib prefix")
set(PYTHON_MODULE_EXTENSION ".so" CACHE INTERNAL "Cross python lib extension")

# Install pybind11
nnfw_find_package(Pybind11 REQUIRED)
if(NOT Pybind11_FOUND)
  message(STATUS "Build onert/python: FAILED (Pybind11 is missing)")
  return()
endif()

# Add the nnfw api and Pybind11 header
include_directories(../../api/include ../../api/src ../../core/include)
include_directories(${CMAKE_BINARY_DIR}/../../../externals/PYBIND11/include/pybind11/)

# Add the pybind tools include
list(APPEND CMAKE_MODULE_PATH "${CMAKE_BINARY_DIR}/../overlay/share/cmake/pybind11/")
# Add the Python module
pybind11_add_module(nnfw_api_pybind src/nnfw_api_pybind.cc)

target_include_directories(nnfw_api_pybind PRIVATE include ../../api/include)
target_link_libraries(nnfw_api_pybind PRIVATE nnfw-dev)

# Install the Python module
install(TARGETS nnfw_api_pybind DESTINATION lib)
