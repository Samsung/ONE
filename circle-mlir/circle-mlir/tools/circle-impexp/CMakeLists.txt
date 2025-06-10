set(SRC
  src/circleimpexp.cpp
  src/driver.cpp
)

add_executable(circle_impexp ${SRC})
cir_mlir_static_flags(circle_impexp)
# NOTE cir_mlir_static_flags sets scope as PUBLIC, so have to give also here
target_link_libraries(circle_impexp PUBLIC cirmlir_dialect)
target_link_libraries(circle_impexp PUBLIC cirmlir_import)
target_link_libraries(circle_impexp PUBLIC cirmlir_export)
target_link_libraries(circle_impexp PUBLIC arser)
target_link_libraries(circle_impexp PUBLIC cirmlir_coverage)

if(NOT ENABLE_TEST)
  return()
endif()

set(SRC_TEST
  src/circleimpexp.cpp
  src/circleimpexp.test.cpp
)

GTest_AddTest_Public(circle_impexp_test ${SRC_TEST})
cir_mlir_static_flags(circle_impexp_test)
target_link_libraries(circle_impexp_test PUBLIC cirmlir_dialect)
target_link_libraries(circle_impexp_test PUBLIC cirmlir_import)
target_link_libraries(circle_impexp_test PUBLIC cirmlir_export)
target_link_libraries(circle_impexp_test PUBLIC cirmlir_coverage)
