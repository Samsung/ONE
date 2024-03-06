set(SRCS_EVAL_TESTER Driver.cpp)

add_executable(onert_micro_eval_driver ${SRCS_EVAL_TESTER})

# This variable is needed to separate standalone interpreter libraries from the libraries used in driver
set(READER_SUFFIX "_driver")

add_subdirectory(${NNAS_PROJECT_SOURCE_DIR}/onert-micro/onert-micro ${CMAKE_CURRENT_BINARY_DIR}/onert-micro)

target_include_directories(onert_micro_eval_driver PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/onert_micro/include")
target_link_libraries(onert_micro_eval_driver PUBLIC onert_micro_interpreter)

install(TARGETS onert_micro_eval_driver DESTINATION bin)

message(STATUS "DONE eval driver")

if(NOT ENABLE_ONERT_MICRO_TRAINING)
    return()
endif(NOT ENABLE_ONERT_MICRO_TRAINING)

set(SRCS_EVAL_TRAINING_TESTER TrainingDriver.cpp)

add_executable(onert_micro_training_eval_driver ${SRCS_EVAL_TRAINING_TESTER})

# This variable is needed to separate standalone interpreter libraries from the libraries used in driver
set(READER_SUFFIX "_driver")

target_include_directories(onert_micro_training_eval_driver PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/onert_micro/include")
target_link_libraries(onert_micro_training_eval_driver PUBLIC onert_micro_interpreter)

install(TARGETS onert_micro_training_eval_driver DESTINATION bin)

message(STATUS "DONE training eval driver")
