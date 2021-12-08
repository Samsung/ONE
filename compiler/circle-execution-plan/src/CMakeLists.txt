set(SOURCES
        "${CIRCLE_EXECUTION_PLAN_PAL_DIR}/IScratchpadHelper.h"
        "${CIRCLE_EXECUTION_PLAN_PAL_DIR}/ScratchpadHelperLinux.h"
        "${CIRCLE_EXECUTION_PLAN_PAL_DIR}/ScratchpadHelperMCU.h"
        "${CIRCLE_EXECUTION_PLAN_PAL_DIR}/ScratchpadHelperCMSISNN.h"
        CircleExecutionPlan.cpp
        ExecutionPlanner.cpp
        ExecutionPlanner.h
        )

set(CIRCLE_EXECUTION_PLAN "circle_execution_plan")

add_executable(${CIRCLE_EXECUTION_PLAN} "${SOURCES}")
target_link_libraries(${CIRCLE_EXECUTION_PLAN} foder)
target_link_libraries(${CIRCLE_EXECUTION_PLAN} safemain)
target_link_libraries(${CIRCLE_EXECUTION_PLAN} luci_env)
target_link_libraries(${CIRCLE_EXECUTION_PLAN} luci_import)
target_link_libraries(${CIRCLE_EXECUTION_PLAN} luci_export)
target_link_libraries(${CIRCLE_EXECUTION_PLAN} luci_plan)
target_link_libraries(${CIRCLE_EXECUTION_PLAN} arser)

target_include_directories(${CIRCLE_EXECUTION_PLAN} PUBLIC "${CIRCLE_EXECUTION_PLAN_PAL_DIR}")
