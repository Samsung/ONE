find_package(PythonInterp 3.8 QUIET)
find_package(PythonLibs 3.8 QUIET)

if(NOT ${PYTHONINTERP_FOUND})
  message(STATUS "Build dalgona: FAILED (Python3 is missing)")
  return()
endif()

if(${PYTHON_VERSION_MINOR} LESS 8)
  message(STATUS "Build dalgona: FAILED (Install Python version higher than or equal to 3.8)")
  return()
endif()

nnas_find_package(Pybind11)
if(NOT Pybind11_FOUND)
  message(STATUS "Build dalgona: FAILED (Pybind11 is missing)")
  return()
endif(NOT Pybind11_FOUND)

set(DRIVER "driver/Driver.cpp")

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(dalgona ${DRIVER} ${SOURCES})
target_include_directories(dalgona PRIVATE include)
target_include_directories(dalgona PRIVATE ${PYTHON_INCLUDE_DIRS})
target_include_directories(dalgona PRIVATE ${Pybind11_INCLUDE_DIRS})

target_link_libraries(dalgona INTERFACE pybind11::embed)
target_link_libraries(dalgona PRIVATE ${PYTHON_LIBRARIES})
target_link_libraries(dalgona PRIVATE arser)
target_link_libraries(dalgona PRIVATE safemain)
target_link_libraries(dalgona PRIVATE foder)
target_link_libraries(dalgona PRIVATE luci_import)
target_link_libraries(dalgona PRIVATE luci_interpreter)
target_link_libraries(dalgona PRIVATE dio_hdf5)
