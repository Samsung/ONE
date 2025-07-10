set(PYTHON_MODULE_PREFIX "lib")
set(PYTHON_MODULE_EXTENSION ".so")

if(CMAKE_CROSSCOMPILING)
  if(NOT BUILD_PYTHON_BINDING)
    return()
  endif(NOT BUILD_PYTHON_BINDING)

  # FindPythonLibs is deprecated since 3.12, and recommand to use FindPython.
  # But on cross compile, FindPython is not working for target environment
  # For workaround, use PythonLibs
  find_package(PythonLibs REQUIRED)

  # Disable pybind11 python search mechanism
  set(PYTHONLIBS_FOUND TRUE CACHE INTERNAL "")

  nnfw_find_package(Pybind11 REQUIRED)
else(CMAKE_CROSSCOMPILING)
  nnfw_find_package(Pybind11 QUIET)
  if(NOT Pybind11_FOUND)
    message(STATUS "Build onert/python: FAILED (Pybind11 is missing)")
    return()
  endif()
endif(CMAKE_CROSSCOMPILING)

# Add the Python module
file(GLOB_RECURSE NNFW_API_PYBIND_SOURCES "src/*.cc")
pybind11_add_module(nnfw_api_pybind ${NNFW_API_PYBIND_SOURCES})
target_include_directories(nnfw_api_pybind PRIVATE include)
target_link_libraries(nnfw_api_pybind PRIVATE nnfw-dev)
target_link_libraries(nnfw_api_pybind PRIVATE nnfw_common nnfw_coverage)

# Install the Python module
install(TARGETS nnfw_api_pybind DESTINATION ${ONERT_INSTALL_APIDIR})
