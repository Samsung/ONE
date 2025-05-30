# NOTE find_package will try to use at least python3.8 as follows depending on platform version
#   Ubuntu18.04; explictly installed python3.8 (default is python3.6)
#   Ubuntu20.04; default python3.8
#   Ubuntu22.04; default python3.10
#   Ubuntu24.04; default python3.12
#   refer https://github.com/Samsung/ONE/issues/9962
# NOTE Require same python version of common-artifacts
# find python 3.8 or above
find_package(Python 3.8 EXACT COMPONENTS Interpreter Development QUIET)
if(NOT Python_FOUND)
  find_package(Python 3.8 COMPONENTS Interpreter Development QUIET)
endif()

if(NOT Python_Development_FOUND)
  message(STATUS "Build dalgona: FAILED (Python3 development package is missing)")
  return()
endif()

# NOTE assume only use 3.8.x or 3.10.x or 3.12.x
if((Python_VERSION VERSION_GREATER_EQUAL 3.8) AND (Python_VERSION VERSION_LESS 3.9))
  set(PYTHON_VERSION_MINOR 8)
elseif((Python_VERSION VERSION_GREATER_EQUAL 3.10) AND (Python_VERSION VERSION_LESS 3.11))
  set(PYTHON_VERSION_MINOR 10)
elseif((Python_VERSION VERSION_GREATER_EQUAL 3.12) AND (Python_VERSION VERSION_LESS 3.13))
  set(PYTHON_VERSION_MINOR 12)
else()
  # TODO support more
  message(STATUS "Build dalgona: FAILED (Unsupported python: ${Python_VERSION})")
  return()
endif()

set(PYTHON_INCLUDE_DIRS ${Python_INCLUDE_DIRS})
set(PYTHON_LIBRARIES ${Python_LIBRARIES})

nnas_find_package(Pybind11)
if(NOT Pybind11_FOUND)
  message(STATUS "Build dalgona: FAILED (Pybind11 is missing)")
  return()
endif(NOT Pybind11_FOUND)

set(DRIVER "driver/Driver.cpp")

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_compile_options(-fvisibility=hidden)

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
target_link_libraries(dalgona PRIVATE nncc_common)

install(TARGETS dalgona DESTINATION bin)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

# dalgona is executable, so we do not link it to the test.
# Instead, we use TEST_SOURCES to specify sources used for tests.
set(TEST_SOURCES
    "src/StringUtils.cpp"
    "src/RandomUtils.cpp")

nnas_find_package(GTest REQUIRED)
GTest_AddTest(dalgona_unit_test ${TESTS} ${TEST_SOURCES})
target_include_directories(dalgona_unit_test PRIVATE src)
target_link_libraries(dalgona_unit_test luci_lang)
