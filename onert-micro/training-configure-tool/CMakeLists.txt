message(STATUS "START Training Config Tool")

add_definitions(-DOM_MEMORY_ESTIMATE)

set(TRAIN_CONFIG_TOOL_SRC
        TrainingConfigureTool.cpp
        src/SparseBackpropagationHandler.cpp
        src/TensorRankSparseBackpropagationHandler.cpp
        src/TrainingConfigureFileHandler.cpp
        src/TrainingDriverHandler.cpp
        src/SparseBackpropagationHelper.cpp)

add_executable(train_config_tool ${TRAIN_CONFIG_TOOL_SRC})

# This variable is needed to separate standalone interpreter libraries from the libraries used in tool
set(CUSTOM_OM_SUFFIX "_train_config_tool")
add_subdirectory(${NNAS_PROJECT_SOURCE_DIR}/onert-micro/onert-micro ${CMAKE_CURRENT_BINARY_DIR}/onert-micro)

target_include_directories(train_config_tool PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/onert_micro/include")
target_include_directories(train_config_tool PUBLIC "include")
target_link_libraries(train_config_tool PUBLIC onert_micro_interpreter)
target_include_directories(train_config_tool PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/onert_micro/include")
target_link_libraries(train_config_tool PUBLIC onert_micro_training_interpreter)

install(TARGETS train_config_tool DESTINATION bin)

message(STATUS "DONE Training Config Tool")
