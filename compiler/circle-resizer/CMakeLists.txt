list(APPEND CIRCLE_RESIZER_SOURCES src/Shape.cpp)
list(APPEND CIRCLE_RESIZER_SOURCES src/ShapeParser.cpp)
list(APPEND CIRCLE_RESIZER_SOURCES src/CircleResizer.cpp)

add_library(circle_resizer_core STATIC "${CIRCLE_RESIZER_SOURCES}")

target_include_directories(circle_resizer_core PUBLIC include)

target_link_libraries(circle_resizer_core PRIVATE mio_circle08)
target_link_libraries(circle_resizer_core PRIVATE logo_core)
target_link_libraries(circle_resizer_core PRIVATE luci_pass)
target_link_libraries(circle_resizer_core PRIVATE luci_lang)
target_link_libraries(circle_resizer_core PRIVATE luci_export)
target_link_libraries(circle_resizer_core PRIVATE luci_import)
target_link_libraries(circle_resizer_core PRIVATE logo)
target_link_libraries(circle_resizer_core PRIVATE luci_log)

add_executable(circle_resizer src/main.cpp)
target_link_libraries(circle_resizer PRIVATE circle_resizer_core)

set_target_properties(circle_resizer PROPERTIES 
  OUTPUT_NAME "circle-resizer" 
)

install(TARGETS circle_resizer DESTINATION bin)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

list(APPEND CIRCLE_RESIZER_TEST_SOURCES tests/ShapeParser.test.cpp)
list(APPEND CIRCLE_RESIZER_TEST_SOURCES tests/CircleResizer.test.cpp)

nnas_find_package(GTest REQUIRED)
GTest_AddTest(circle_resizer_unit_test ${CIRCLE_RESIZER_TEST_SOURCES})
target_link_libraries(circle_resizer_unit_test circle_resizer_core)

get_target_property(ARTIFACTS_PATH testDataGenerator BINARY_DIR)
set_tests_properties(circle_resizer_unit_test
                     PROPERTIES
                     ENVIRONMENT "ARTIFACTS_PATH=${ARTIFACTS_PATH}")
