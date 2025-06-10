# NOTE find_package will try to use at least python3.8 as follows depending on platform version
#   Ubuntu18.04; explictly installed python3.8 (default is python3.6)
#   Ubuntu20.04; default python3.8
#   Ubuntu22.04; default python3.10
#   Ubuntu24.04; explictly installed python3.8 (default is python3.12)
#   refer https://github.com/Samsung/ONE/issues/9962
if(CMAKE_VERSION VERSION_LESS 3.12)
  find_package(PythonInterp 3.8 QUIET)
  find_package(PythonLibs 3.8 QUIET)

  if(NOT ${PYTHONINTERP_FOUND})
    message(STATUS "Build circle-resizer: FAILED (Python3 is missing)")
    return()
  endif()

  if(${PYTHON_VERSION_MINOR} LESS 8)
    message(STATUS "Build circle-resizer: FAILED (Install Python version higher than or equal to 3.8)")
    return()
  endif()
else()
  find_package(Python 3.8 EXACT COMPONENTS Development QUIET)
  if(NOT Python_FOUND)
    find_package(Python 3.8 COMPONENTS Development QUIET)
  endif()

  # Require same python version of common-artifacts
  if(Python_VERSION VERSION_GREATER_EQUAL 3.12)
    message(STATUS "Build dalgona: FALSE (Python version 3.12 or higher is not supported yet)")
    return()
  endif()
  if(Python_VERSION VERSION_LESS 3.8)
    message(STATUS "Build dalgona: FAILED (Install Python version 3.8 or 3.10)")
    return()
  endif()

  if(NOT Python_Development_FOUND)
    message(STATUS "Build dalgona: FAILED (Python3 development package is missing)")
    return()
  endif()

  set(PYTHON_INCLUDE_DIRS ${Python_INCLUDE_DIRS})
  set(PYTHON_LIBRARIES ${Python_LIBRARIES})
endif()

nnas_find_package(Pybind11)
if(NOT Pybind11_FOUND)
  message(STATUS "circle-resizer: FAILED (Pybind11 is missing)")
  return()
endif()

add_library(circle_resizer_python_api SHARED src/CircleResizerBindings.cpp)
target_link_libraries(circle_resizer_python_api PRIVATE circle_resizer_core)

set_target_properties(circle_resizer_python_api PROPERTIES CXX_VISIBILITY_PRESET "hidden"
                                         CUDA_VISIBILITY_PRESET "hidden")
set_target_properties(circle_resizer_python_api PROPERTIES PREFIX "")

target_include_directories(circle_resizer_python_api PRIVATE ${PYTHON_INCLUDE_DIRS})
target_include_directories(circle_resizer_python_api PRIVATE ${Pybind11_INCLUDE_DIRS})

install(TARGETS circle_resizer_python_api DESTINATION lib)
