nnas_find_package(HDF5 COMPONENTS STATIC QUIET)

if(NOT HDF5_FOUND)
  message(STATUS "Build minmax_embedder: FAILED (missing HDF5)")
  return()
endif(NOT HDF5_FOUND)

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE DRIVER "driver/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${DRIVER})
list(REMOVE_ITEM SOURCES ${TESTS})

#
# Library
#
add_library(minmax_embedder "${SOURCES}")
target_include_directories(minmax_embedder PUBLIC ${HDF5_INCLUDE_DIRS})
target_include_directories(minmax_embedder PRIVATE include)

target_link_libraries(minmax_embedder ${HDF5_CXX_LIBRARIES})
target_link_libraries(minmax_embedder safemain)
target_link_libraries(minmax_embedder loco)
target_link_libraries(minmax_embedder luci_import)
target_link_libraries(minmax_embedder luci_service)
target_link_libraries(minmax_embedder luci_pass)
target_link_libraries(minmax_embedder luci_export)
target_link_libraries(minmax_embedder luci_env)
target_link_libraries(minmax_embedder arser)
target_link_libraries(minmax_embedder vconone)

install(TARGETS minmax_embedder DESTINATION lib)
install(DIRECTORY include/ DESTINATION include
        FILES_MATCHING PATTERN "*.h")

#
# Driver
#
add_executable(minmax_embedder_driver "${DRIVER}")
target_link_libraries(minmax_embedder_driver minmax_embedder)
target_include_directories(minmax_embedder_driver PRIVATE include)
set_target_properties(minmax_embedder_driver PROPERTIES OUTPUT_NAME minmax_embedder)

install(TARGETS minmax_embedder DESTINATION bin)

#
# GTest
#
if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

GTest_AddTest(minmax_embedder_test ${TESTS})
target_include_directories(minmax_embedder_test PRIVATE include)
target_link_libraries(minmax_embedder_test minmax_embedder)
