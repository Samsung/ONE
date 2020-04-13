file(GLOB_RECURSE SOURCES "src/*.cc")

add_library(neurun_core SHARED ${SOURCES})
set_target_properties(neurun_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(neurun_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(neurun_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(neurun_core PUBLIC nnfw_lib_misc)
target_link_libraries(neurun_core PRIVATE nnfw_lib_cker)
target_link_libraries(neurun_core PRIVATE nnfw_common)
target_link_libraries(neurun_core PRIVATE nnfw_coverage)
target_link_libraries(neurun_core PRIVATE dl ${LIB_PTHREAD})

if(ENVVAR_NEURUN_CONFIG)
  target_compile_definitions(neurun_core PRIVATE ENVVAR_FOR_DEFAULT_CONFIG)
endif(ENVVAR_NEURUN_CONFIG)

install(TARGETS neurun_core DESTINATION lib)
