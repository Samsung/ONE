set(SOURCES
        src/CircleMemoryPlan.cpp
        src/MemoryPlanner.cpp
        src/MemoryPlanner.h
        )

add_executable(circle_memory_plan "${SOURCES}")
target_link_libraries(circle_memory_plan foder)
target_link_libraries(circle_memory_plan safemain)
target_link_libraries(circle_memory_plan luci_env)
target_link_libraries(circle_memory_plan luci_import)
target_link_libraries(circle_memory_plan luci_export)
target_link_libraries(circle_memory_plan arser)

install(TARGETS circle_memory_plan DESTINATION bin)

