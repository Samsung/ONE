# NOTE find_package will try to use at least python3.8 as follows depending on platform version
#   Ubuntu18.04; explictly installed python3.8 (default is python3.6)
#   Ubuntu20.04; default python3.8
#   Ubuntu22.04; default python3.10
#   refer https://github.com/Samsung/ONE/issues/9962
find_package(Python3 3.8 QUIET COMPONENTS Interpreter Development)

if(NOT Python3_FOUND)
  message(STATUS "Build dalgona: FAILED (Python3 is missing)")
  return()
endif()

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import sys; print(sys.version_info.minor)"
  OUTPUT_VARIABLE PYTHON_VERSION_MINOR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

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
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_compile_options(-fvisibility=hidden)

add_executable(dalgona ${DRIVER} ${SOURCES})
target_include_directories(dalgona PRIVATE include)
target_include_directories(dalgona PRIVATE ${Python3_INCLUDE_DIRS})
target_include_directories(dalgona PRIVATE ${Pybind11_INCLUDE_DIRS})

target_link_libraries(dalgona INTERFACE pybind11::embed)
target_link_libraries(dalgona PRIVATE Python3::Python)
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
