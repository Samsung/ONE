file(GLOB_RECURSE SOURCES "src/*.cc")

add_library(onert_core SHARED ${SOURCES})
set_target_properties(onert_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(onert_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(onert_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(onert_core PUBLIC nnfw_lib_misc half)
target_link_libraries(onert_core PRIVATE nnfw_lib_cker)
target_link_libraries(onert_core PRIVATE nnfw_common)
target_link_libraries(onert_core PRIVATE nnfw_coverage)
target_link_libraries(onert_core PRIVATE dl ${LIB_PTHREAD})
target_link_libraries(onert_core INTERFACE ruy_instrumentation)

if(ENVVAR_ONERT_CONFIG)
  target_compile_definitions(onert_core PRIVATE ENVVAR_FOR_DEFAULT_CONFIG)
endif(ENVVAR_ONERT_CONFIG)

install(TARGETS onert_core LIBRARY DESTINATION lib)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
        DESTINATION "include/onert"
        FILES_MATCHING PATTERN "*.h" PATTERN "*.lst"
        )
