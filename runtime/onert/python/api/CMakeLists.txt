# CMakeLists.txt

# refer to https://github.com/Samsung/ONE/issues/11368

# Tell pybind11 where the target python installation is
#
# FindPythonLibs is deprecated since 3.12, and recommand to use FindPython.
# But on cross compile, FindPython is not working for target environment
# For workaround, use PythonLibs
find_package(PythonLibs REQUIRED)
set(PYTHON_MODULE_PREFIX "lib" CACHE INTERNAL "Cross python lib prefix")
set(PYTHON_MODULE_EXTENSION ".so" CACHE INTERNAL "Cross python lib extension")

# Disable pybind11 python search mechanism
set(PYTHONLIBS_FOUND TRUE CACHE INTERNAL "")

# Install pybind11
nnfw_find_package(Pybind11 REQUIRED)
if(NOT Pybind11_FOUND)
  message(STATUS "Build onert/python: FAILED (Pybind11 is missing)")
  return()
endif()

# Add the Python module
file(GLOB_RECURSE NNFW_API_PYBIND_SOURCES "src/*.cc")
pybind11_add_module(nnfw_api_pybind ${NNFW_API_PYBIND_SOURCES})
target_include_directories(nnfw_api_pybind PRIVATE include)
target_link_libraries(nnfw_api_pybind PRIVATE nnfw-dev)

# Install the Python module
install(TARGETS nnfw_api_pybind DESTINATION lib)
