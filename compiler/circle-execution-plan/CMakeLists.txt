set(SOURCES
        src/CircleExecutionPlan.cpp
        src/ExecutionPlanner.cpp
        src/ExecutionPlanner.h
        )

file(GLOB_RECURSE TESTS "src/*.test.cpp" "src/ExecutionPlanner.cpp")

add_executable(circle_execution_plan "${SOURCES}")
target_link_libraries(circle_execution_plan foder)
target_link_libraries(circle_execution_plan safemain)
target_link_libraries(circle_execution_plan luci_env)
target_link_libraries(circle_execution_plan luci_import)
target_link_libraries(circle_execution_plan luci_export)
target_link_libraries(circle_execution_plan luci_plan)
target_link_libraries(circle_execution_plan arser)

install(TARGETS circle_execution_plan DESTINATION bin)

if(NOT ENABLE_TEST)
    return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

GTest_AddTest(execution-planner-test ${TESTS})
target_include_directories(execution-planner-test PRIVATE src)
target_link_libraries(execution-planner-test luci_plan)
target_link_libraries(execution-planner-test luci_env)
