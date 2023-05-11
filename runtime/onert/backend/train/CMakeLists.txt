set(LIB_ONERT_BACKEND_TRAIN onert_backend_train)

file(GLOB_RECURSE SOURCES "*.cc")

add_library(${LIB_ONERT_BACKEND_TRAIN} SHARED ${SOURCES})

target_link_libraries(${LIB_ONERT_BACKEND_TRAIN} PRIVATE ${LIB_ONERT_BACKEND_CPU})
target_link_libraries(${LIB_ONERT_BACKEND_TRAIN} PRIVATE onert_core)
target_link_libraries(${LIB_ONERT_BACKEND_TRAIN} PRIVATE nnfw_lib_cker nnfw_lib_misc)
target_link_libraries(${LIB_ONERT_BACKEND_TRAIN} PRIVATE nnfw_common)
target_link_libraries(${LIB_ONERT_BACKEND_TRAIN} PRIVATE nnfw_coverage)

set_target_properties(${LIB_ONERT_BACKEND_TRAIN} PROPERTIES OUTPUT_NAME backend_train)

if(CMAKE_BUILD_TYPE_LC STREQUAL "release")
  add_custom_command(TARGET ${LIB_ONERT_BACKEND_TRAIN} POST_BUILD
                     COMMAND ${CMAKE_STRIP} "--strip-unneeded" $<TARGET_FILE_NAME:${LIB_ONERT_BACKEND_TRAIN}>)
endif()

install(TARGETS ${LIB_ONERT_BACKEND_TRAIN} DESTINATION lib)
