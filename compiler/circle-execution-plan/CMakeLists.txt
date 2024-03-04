nnas_find_package(Jsoncpp)
if(NOT Jsoncpp_FOUND)
    message(STATUS "Build circle-execution-plan: FAILED (missing jsoncpp)")
    return()
endif(NOT Jsoncpp_FOUND)

set(SOURCES
        pal/IScratchpadHelper.h
        pal/ScratchpadHelperLinux.h
        pal/ScratchpadHelperMCU.h
        pal/ScratchpadHelperCMSISNN.h
        pal/TargetPlatform.h
        src/CircleExecutionPlan.cpp
        src/ExecutionPlanner.cpp
        src/ExecutionPlanner.h
        )

add_executable(circle_execution_plan "${SOURCES}")
target_include_directories(circle_execution_plan PRIVATE ${Jsoncpp_INCLUDE_DIRS})

target_link_libraries(circle_execution_plan ${Jsoncpp_STATIC_LIB})
target_link_libraries(circle_execution_plan foder)
target_link_libraries(circle_execution_plan safemain)
target_link_libraries(circle_execution_plan luci_env)
target_link_libraries(circle_execution_plan luci_import)
target_link_libraries(circle_execution_plan luci_export)
target_link_libraries(circle_execution_plan luci_plan)
target_link_libraries(circle_execution_plan arser)
target_link_libraries(circle_execution_plan luci_log)

target_include_directories(circle_execution_plan PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/pal")
install(TARGETS circle_execution_plan DESTINATION bin)

set(SOURCES_2
        src/CircleTrainingConfigure.cpp
        src/GenerateTrainingGraph.cpp
        )

add_executable(circle_training_configure "${SOURCES_2}")
target_include_directories(circle_training_configure PRIVATE ${Jsoncpp_INCLUDE_DIRS})

target_link_libraries(circle_training_configure ${Jsoncpp_STATIC_LIB})
target_link_libraries(circle_training_configure foder)
target_link_libraries(circle_training_configure safemain)
target_link_libraries(circle_training_configure luci_env)
target_link_libraries(circle_training_configure luci_import)
target_link_libraries(circle_training_configure luci_export)
target_link_libraries(circle_training_configure luci_plan)
target_link_libraries(circle_training_configure arser)
target_link_libraries(circle_training_configure luci_log)

target_include_directories(circle_training_configure PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/pal")
install(TARGETS circle_training_configure DESTINATION bin)
