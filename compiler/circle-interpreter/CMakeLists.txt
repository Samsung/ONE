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
    message(STATUS "Build circle-interpreter: FAILED (Python3 is missing)")
    return()
  endif()

  if(${PYTHON_VERSION_MINOR} LESS 8)
    message(STATUS "Build circle-interpreter: FAILED (Install Python version higher than or equal to 3.8)")
    return()
  endif()
else()
  find_package(Python 3.8 EXACT COMPONENTS Development Interpreter QUIET)
  if(NOT Python_FOUND)
    find_package(Python 3.8 COMPONENTS Development Interpreter QUIET)
  endif()

  if(NOT Python_Development_FOUND)
    message(STATUS "Build circle-interpreter: FAILED (Python3 development package is missing)")
    return()
  endif()

  set(PYTHON_INCLUDE_DIRS ${Python_INCLUDE_DIRS})
  set(PYTHON_LIBRARIES ${Python_LIBRARIES})
endif()

# nanobind depends on robin-map
nnas_find_package(RobinMap REQUIRED)
if(NOT RobinMap_FOUND)
  message(STATUS "Build circle-interpreter: FAILED (RobinMap is missing)")
  return()
endif()

nnas_find_package(Nanobind)
if(NOT Nanobind_FOUND)
  message(STATUS "Build circle-interpreter: FAILED (Nanobind is missing)")
  return()
endif(NOT Nanobind_FOUND)

nanobind_add_module(my_ext src/my_ext.cpp)
install(TARGETS my_ext DESTINATION lib)

set(INTERPRETER
      src/CircleInterpreter.cpp
   )

add_executable(circle-interpreter ${INTERPRETER})
target_link_libraries(circle-interpreter PRIVATE arser)
target_link_libraries(circle-interpreter PRIVATE loco)
target_link_libraries(circle-interpreter PRIVATE luci_import)
target_link_libraries(circle-interpreter PRIVATE luci_interpreter)
target_link_libraries(circle-interpreter PRIVATE safemain)
target_link_libraries(circle-interpreter PRIVATE vconone)

install(TARGETS circle-interpreter DESTINATION bin)
