find_package(PythonInterp 3.8 QUIET)
find_package(PythonLibs 3.8 QUIET)

if(NOT ${PYTHONINTERP_FOUND})
  message(STATUS "Build dalgona: FALSE (Python3 is missing)")
  return()
endif()

if(${PYTHON_VERSION_MINOR} LESS 6)
  message(STATUS "Build dalgona: FALSE (You need to install Python version higher than 3.6)")
  return()
endif()

message(STATUS "Python libraries")
message(STATUS ${PYTHON_LIBRARIES})
set (Python_INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS})
set (Python_LIBRARIES ${PYTHON_LIBRARIES})

nnas_find_package(Pybind11)
if(NOT Pybind11_FOUND)
  message(STATUS "Build dalgona: FAILED (missing Pybind11)")
  return()
endif(NOT Pybind11_FOUND)

add_subdirectory(tests)

set(DRIVER "driver/Driver.cpp")

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(dalgona ${DRIVER} ${SOURCES})
target_include_directories(dalgona PRIVATE include)
target_include_directories(dalgona PRIVATE ${Python_INCLUDE_DIRS})
target_include_directories(dalgona PRIVATE ${Pybind11_INCLUDE_DIRS})

target_link_libraries(dalgona INTERFACE pybind11::embed)
target_link_libraries(dalgona PRIVATE ${Python_LIBRARIES})
target_link_libraries(dalgona PRIVATE arser)
target_link_libraries(dalgona PRIVATE safemain)
target_link_libraries(dalgona PRIVATE luci_import)
target_link_libraries(dalgona PRIVATE luci_interpreter)
target_link_libraries(dalgona PRIVATE dio_hdf5)
