set(LIB_ONERT_BACKEND_TRAINING onert_backend_training)

nnfw_find_package(Ruy REQUIRED)

file(GLOB_RECURSE SOURCES "*.cc")

add_library(${LIB_ONERT_BACKEND_TRAINING} SHARED ${SOURCES})

target_link_libraries(${LIB_ONERT_BACKEND_TRAINING} PRIVATE nnfw_lib_cker nnfw_lib_misc)
target_link_libraries(${LIB_ONERT_BACKEND_TRAINING} PRIVATE onert_core)
target_link_libraries(${LIB_ONERT_BACKEND_TRAINING} PRIVATE nnfw_common)
target_link_libraries(${LIB_ONERT_BACKEND_TRAINING} PRIVATE nnfw_coverage)
target_link_libraries(${LIB_ONERT_BACKEND_TRAINING} PRIVATE ruy)
target_link_libraries(${LIB_ONERT_BACKEND_TRAINING} INTERFACE ruy_instrumentation)
target_link_libraries(${LIB_ONERT_BACKEND_TRAINING} PRIVATE ndarray)

set_target_properties(${LIB_ONERT_BACKEND_TRAINING} PROPERTIES OUTPUT_NAME backend_training)

if(CMAKE_BUILD_TYPE_LC STREQUAL "release")
  add_custom_command(TARGET ${LIB_ONERT_BACKEND_TRAINING} POST_BUILD
                     COMMAND ${CMAKE_STRIP} "--strip-unneeded" $<TARGET_FILE_NAME:${LIB_ONERT_BACKEND_TRAINING}>)
endif()

install(TARGETS ${LIB_ONERT_BACKEND_TRAINING} DESTINATION lib)
