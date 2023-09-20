# CMakeLists.txt

# refer https://github.com/Samsung/ONE/issues/11368

set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY "Set to find packages only in cross root path by force")
find_package(Python COMPONENTS Interpreter Development)

# Tell pybind11 where the target python installation is
set(PYTHON_INCLUDE_DIRS ${Python_INCLUDE_DIRS} CACHE INTERNAL "Cross python include path")
set(PYTHON_LIBRARIES ${Python_LIBRARIES} CACHE INTERNAL "Cross python include path")
set(PYTHON_MODULE_PREFIX "lib" CACHE INTERNAL "Cross python lib prefix")
set(PYTHON_MODULE_EXTENSION ".so" CACHE INTERNAL "Cross python lib extension")

# Disable pybind11 python search mechanism
set(PYTHONLIBS_FOUND TRUE CACHE INTERNAL "")

# install pybind11
nnfw_find_package(Pybind11 REQUIRED)
if(NOT Pybind11_FOUND)
  message(STATUS "Build onert/python: FAILED (Pybind11 is missing)")
  return()
endif()

# Add the Python module
file(GLOB_RECURSE NNFW_API_PYBIND_SOURCES "src/*.cc")
pybind11_add_module(nnfw_api_pybind ${NNFW_API_PYBIND_SOURCES})

# Add the nnfw api and Pybind11 header
target_include_directories(nnfw_api_pybind PRIVATE ${pybind11_SOURCE_DIR}/include/pybind11)
target_include_directories(nnfw_api_pybind PRIVATE include ../../api/include)
target_link_libraries(nnfw_api_pybind PRIVATE nnfw-dev)

# Install the Python module
install(TARGETS nnfw_api_pybind DESTINATION lib)
