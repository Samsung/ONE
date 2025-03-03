# FindPythonLibs is deprecated since 3.12, and recommand to use FindPython.
# But on cross compile, FindPython is not working for target environment
# For workaround, use PythonLibs
# refer to https://github.com/Samsung/ONE/issues/11368 for more details
find_package(PythonLibs REQUIRED)
set(PYTHON_MODULE_PREFIX "lib" CACHE INTERNAL "Cross python lib prefix")
set(PYTHON_MODULE_EXTENSION ".so" CACHE INTERNAL "Cross python lib extension")

# Disable pybind11 python search mechanism
set(PYTHONLIBS_FOUND TRUE CACHE INTERNAL "")

nnas_find_package(Pybind11)
if(NOT Pybind11_FOUND)
  message(STATUS "circle-resizer: FAILED (Pybind11 is missing)")
  return()
endif()

list(APPEND CIRCLE_RESIZER_PYTHON_API src/Shape.cpp)

add_library(circle_resizer_python_api MODULE ${CIRCLE_RESIZER_PYTHON_API})
target_link_libraries(circle_resizer_python_api INTERFACE pybind11::module)

set_target_properties(circle_resizer_python_api PROPERTIES CXX_VISIBILITY_PRESET "hidden"
                                         CUDA_VISIBILITY_PRESET "hidden")

target_include_directories(circle_resizer_python_api PRIVATE include)

target_include_directories(circle_resizer_python_api PRIVATE ${PYTHON_INCLUDE_DIRS})
target_include_directories(circle_resizer_python_api PRIVATE ${Pybind11_INCLUDE_DIRS})
# target_link_libraries(circle_resizer_python_api PRIVATE ${PYTHON_LIBRARIES})
target_link_libraries(circle_resizer_python_api PRIVATE circle_resizer_core)

# Install the Python module
install(TARGETS circle_resizer_python_api DESTINATION lib)
