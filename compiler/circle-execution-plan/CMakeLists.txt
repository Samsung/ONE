set(SOURCES
        src/CircleExecutionPlan.cpp
        src/ExecutionPlanner.cpp
        src/ExecutionPlanner.h
        )

add_executable(circle_execution_plan "${SOURCES}")
target_link_libraries(circle_execution_plan foder)
target_link_libraries(circle_execution_plan safemain)
target_link_libraries(circle_execution_plan luci_env)
target_link_libraries(circle_execution_plan luci_import)
target_link_libraries(circle_execution_plan luci_export)
target_link_libraries(circle_execution_plan luci_plan)
target_link_libraries(circle_execution_plan arser)

install(TARGETS circle_execution_plan DESTINATION bin)
