if(NOT BUILD_NPUD)
  return()
endif(NOT BUILD_NPUD)

set(NPUD_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(core)
add_subdirectory(tests)
