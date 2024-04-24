nnas_find_package(FlatBuffers EXACT 23.5.26 QUIET)

if(NOT FlatBuffers_FOUND)
  message(STATUS "Build exo: FALSE (missing FlatBuffers)")
  return()
endif(NOT FlatBuffers_FOUND)

nnas_find_package(TensorFlowSource EXACT 1.14 QUIET)

if(NOT TensorFlowSource_FOUND)
  message(STATUS "Build exo: FALSE (missing TensorFlowSource)")
  return()
endif(NOT TensorFlowSource_FOUND)

message(STATUS "Build exo: TRUE")

set(TFLITE_SCHEMA_DIR "${TensorFlowSource_DIR}/tensorflow/lite/schema")
set(CIRCLE_SCHEMA_DIR "${NNAS_PROJECT_SOURCE_DIR}/res/CircleSchema/0.3")

FlatBuffers_Target(exo_tflite_fbs
  OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/gen"
  SCHEMA_DIR "${TFLITE_SCHEMA_DIR}"
  SCHEMA_FILES schema.fbs
)

FlatBuffers_Target(exo_circle_fbs
  OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/gen"
  SCHEMA_DIR "${CIRCLE_SCHEMA_DIR}"
  SCHEMA_FILES circle_schema.fbs
)

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(exo SHARED ${SOURCES})
target_include_directories(exo PUBLIC include)
target_include_directories(exo PRIVATE src)
target_link_libraries(exo PUBLIC exo_tflite_fbs)
target_link_libraries(exo PUBLIC exo_circle_fbs)
target_link_libraries(exo PUBLIC loco)
target_link_libraries(exo PRIVATE pepper_str)
target_link_libraries(exo PRIVATE pepper_strcast)
target_link_libraries(exo PRIVATE locoex_customop)
target_link_libraries(exo PRIVATE locop)
target_link_libraries(exo PRIVATE hermes_std)
target_link_libraries(exo PRIVATE logo)
target_link_libraries(exo PRIVATE oops)
install(TARGETS exo DESTINATION lib)

# Let's apply nncc common compile options
#
# NOTE This will enable strict compilation (warnings as error).
#      Please refer to the top-level CMakeLists.txt for details
target_link_libraries(exo PRIVATE nncc_common)

if (NOT ENABLE_TEST)
  return()
endif (NOT ENABLE_TEST)

# Google Test is mandatory for internal testing
nnas_find_package(GTest REQUIRED)

GTest_AddTest(exo_test ${TESTS})
target_include_directories(exo_test PRIVATE src)
target_link_libraries(exo_test pepper_str)
target_link_libraries(exo_test exo)
target_link_libraries(exo_test hermes_std)
target_link_libraries(exo_test logo)
target_link_libraries(exo_test oops)
target_link_libraries(exo_test locoex_customop)
