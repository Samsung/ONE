file(GLOB_RECURSE SOURCES "src/*.cc")
file(GLOB_RECURSE TESTS "*.test.cc")
list(REMOVE_ITEM SOURCES ${TESTS})

if(NOT BUILD_MINMAX_H5DUMPER)
  file(GLOB_RECURSE SRC_TO_REMOVE "src/dumper/h5/*.cc")
  list(REMOVE_ITEM SOURCES ${SRC_TO_REMOVE})
  file(GLOB_RECURSE SRC_TO_REMOVE "src/exec/MinMaxRecorder.cc")
  list(REMOVE_ITEM SOURCES ${SRC_TO_REMOVE})
endif(NOT BUILD_MINMAX_H5DUMPER)

add_library(onert_core SHARED ${SOURCES})
set_target_properties(onert_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# NOTE
# We publish public headers into developer package.
# To avoid mistake using private header in public header, do not define
# private target_include_directories scope for src/ directory.
target_include_directories(onert_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Flatbuffer for loader
nnfw_find_package(FlatBuffers EXACT 2.0 REQUIRED)
target_link_libraries(onert_core PRIVATE flatbuffers::flatbuffers)
target_link_libraries(onert_core PRIVATE circle_schema)

target_link_libraries(onert_core PRIVATE jsoncpp half)
target_link_libraries(onert_core PRIVATE nnfw_lib_misc nnfw_lib_cker)
target_link_libraries(onert_core PRIVATE nnfw_common)
target_link_libraries(onert_core PRIVATE nnfw_coverage)
target_link_libraries(onert_core PRIVATE dl ${LIB_PTHREAD})

# Ruy
nnfw_find_package(Ruy REQUIRED)
target_link_libraries(onert_core PRIVATE ruy)
target_link_libraries(onert_core INTERFACE ruy_instrumentation)

# H5 Minmax Dumper
if(BUILD_MINMAX_H5DUMPER)
  nnfw_find_package(HDF5 REQUIRED)
  target_compile_definitions(onert_core PRIVATE MINMAX_H5DUMPER=1)
  target_include_directories(onert_core PRIVATE ${HDF5_INCLUDE_DIRS})
  target_link_libraries(onert_core PRIVATE ${HDF5_CXX_LIBRARIES})
endif(BUILD_MINMAX_H5DUMPER)

if(CMAKE_BUILD_TYPE_LC STREQUAL "release")
  add_custom_command(TARGET onert_core POST_BUILD
                     COMMAND ${CMAKE_STRIP} "--strip-unneeded" $<TARGET_FILE_NAME:onert_core>)
endif()

# NOTE Below line is added to remove warning for android build
#      It will be removed after android build uses gold linker
if (ANDROID)
  target_link_libraries(onert_core INTERFACE log)
endif (ANDROID)

if(ENVVAR_ONERT_CONFIG)
  target_compile_definitions(onert_core PRIVATE ENVVAR_FOR_DEFAULT_CONFIG)
endif(ENVVAR_ONERT_CONFIG)

install(TARGETS onert_core LIBRARY DESTINATION lib)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
        DESTINATION "include/onert"
        FILES_MATCHING PATTERN "*.h" PATTERN "*.lst"
        )

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

# Unit Tests
set(TEST_ONERT_CORE test_onert_core)

add_executable(${TEST_ONERT_CORE} ${TESTS})

target_link_libraries(${TEST_ONERT_CORE} onert_core)
# Requires linking nnfw_coverage: check header coverage
target_link_libraries(${TEST_ONERT_CORE} nnfw_coverage)
target_link_libraries(${TEST_ONERT_CORE} gtest gtest_main dl ${LIB_PTHREAD})

add_test(${TEST_ONERT_CORE} ${TEST_ONERT_CORE})
install(TARGETS ${TEST_ONERT_CORE} DESTINATION unittest)
