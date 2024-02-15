#define core path
set(OM_SOURCE_CORE_DIR "${OM_SOURCE_DIR}/core")
set(OM_INCLUDE_CORE_DIR "${OM_INCLUDE_DIR}/core")
#define execute path
set(OM_SOURCE_EXECUTE_DIR "${OM_SOURCE_DIR}/execute")
set(OM_INCLUDE_EXECUTE_DIR "${OM_INCLUDE_DIR}/execute")
#define import path
set(OM_SOURCE_IMPORT_DIR "${OM_SOURCE_DIR}/import")
set(OM_INCLUDE_IMPORT_DIR "${OM_INCLUDE_DIR}/import")
#define optimize path
set(OM_SOURCE_OPTIMIZE_DIR "${OM_SOURCE_DIR}/optimize")
set(OM_INCLUDE_OPTIMIZE_DIR "${OM_INCLUDE_DIR}/optimize")

#OM_Interpreter lib binary name
set(OM_INTERPRETER_LIB "onert_micro_interpreter")
#Core lib binary name
set(OM_CORE_LIB "onert_micro_core${OM_SUFFIX}")
#Execute lib binary name
set(OM_EXECUTE_LIB "onert_micro_execute${OM_SUFFIX}")
#Import lib binary name
set(OM_IMPORT_LIB "onert_micro_import${OM_SUFFIX}")
#Optimize lib binary name
set(OM_OPTIMIZE_LIB "onert_micro_optimize${OM_SUFFIX}")
#PAL lib binary name
set(OM_PAL_LIB "onert_micro_pal${OM_SUFFIX}")

message(STATUS "ONERT MICRO BEGIN")

#build core lib
add_subdirectory(${OM_SOURCE_CORE_DIR})
#build import lib
add_subdirectory(${OM_SOURCE_IMPORT_DIR})
#build execute lib
add_subdirectory(${OM_SOURCE_EXECUTE_DIR})
#build optimize lib
add_subdirectory(${OM_SOURCE_OPTIMIZE_DIR})

target_link_libraries(${OM_CORE_LIB} PUBLIC ${OM_CIRCLE_SCHEMA})
target_link_libraries(${OM_CORE_LIB} PUBLIC ${OM_IMPORT_LIB})
target_link_libraries(${OM_CORE_LIB} PUBLIC ${OM_EXECUTE_LIB})
target_link_libraries(${OM_CORE_LIB} PUBLIC ${OM_OPTIMIZE_LIB})

add_library(${OM_INTERPRETER_LIB} STATIC OMInterpreter.cpp)

target_include_directories(${OM_INTERPRETER_LIB} PUBLIC "${OM_INCLUDE_DIR}")
target_link_libraries(${OM_INTERPRETER_LIB} PUBLIC ${OM_CORE_LIB})

message(STATUS "ONERT MICRO FINISHED")
