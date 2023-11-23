nnas_find_package(TensorFlowSource EXACT 2.8.0 QUIET)
nnas_find_package(TensorFlowGEMMLowpSource EXACT 2.8.0 QUIET)
nnas_find_package(TensorFlowRuySource EXACT 2.8.0 QUIET)
nnas_find_package(NEON2SSESource QUIET)

if(NOT TensorFlowSource_FOUND)
  message(STATUS "Build luci-compute: FAILED (missing TensorFlowSource 2.8.0)")
  return()
endif(NOT TensorFlowSource_FOUND)

if(NOT TensorFlowGEMMLowpSource_FOUND)
  message(STATUS "Build luci-compute: FAILED (missing TensorFlowGEMMLowpSource 2.8.0)")
  return()
endif(NOT TensorFlowGEMMLowpSource_FOUND)

if(NOT TensorFlowRuySource_FOUND)
  message(STATUS "Build luci-compute: FAILED (missing TensorFlowRuySource 2.8.0)")
  return()
endif(NOT TensorFlowRuySource_FOUND)

if(NOT NEON2SSESource_FOUND)
  message(STATUS "Build luci-compute: FAILED (missing NEON2SSESource)")
  return()
endif(NOT NEON2SSESource_FOUND)

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

if (NOT LUCI_LIBRARY_TYPE)
  set(LUCI_LIBRARY_TYPE "SHARED")
endif(NOT LUCI_LIBRARY_TYPE)

add_library(luci_compute ${LUCI_LIBRARY_TYPE} ${SOURCES})
target_include_directories(luci_compute PUBLIC include)
target_include_directories(luci_compute PRIVATE src)
target_include_directories(luci_compute SYSTEM PRIVATE "${TensorFlowSource_DIR}")
target_include_directories(luci_compute SYSTEM PRIVATE "${TensorFlowGEMMLowpSource_DIR}")
target_include_directories(luci_compute SYSTEM PRIVATE "${TensorFlowRuySource_DIR}")
target_include_directories(luci_compute SYSTEM PRIVATE "${NEON2SSESource_DIR}")
target_link_libraries(luci_compute PUBLIC loco)
install(TARGETS luci_compute DESTINATION lib)
install(DIRECTORY include/ DESTINATION include
        FILES_MATCHING PATTERN "*.h")

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

GTest_AddTest(luci_compute_test ${TESTS})
target_include_directories(luci_compute_test PRIVATE src)
target_link_libraries(luci_compute_test luci_compute)
target_include_directories(luci_compute_test SYSTEM PRIVATE "${TensorFlowSource_DIR}")
target_include_directories(luci_compute_test SYSTEM PRIVATE "${TensorFlowGEMMLowpSource_DIR}")
target_include_directories(luci_compute_test SYSTEM PRIVATE "${TensorFlowRuySource_DIR}")
target_include_directories(luci_compute_test SYSTEM PRIVATE "${NEON2SSESource_DIR}")
