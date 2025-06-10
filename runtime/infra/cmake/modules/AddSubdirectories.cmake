function(list_subdirectories OUTPUT_VARIABLE)
  cmake_parse_arguments(ARG "" "" "EXCLUDES" ${ARGN})

  file(GLOB PROJECT_FILES
            RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            "*/CMakeLists.txt")

  foreach(PROJECT_FILE IN ITEMS ${PROJECT_FILES})
    get_filename_component(PROJECT_DIR ${PROJECT_FILE} DIRECTORY)
    list(FIND ARG_EXCLUDES ${PROJECT_DIR} PROJECT_INDEX)
    if(${PROJECT_INDEX} EQUAL -1)
      list(APPEND PROJECT_LIST ${PROJECT_DIR})
    endif(${PROJECT_INDEX} EQUAL -1)
  endforeach(PROJECT_FILE)

  set(${OUTPUT_VARIABLE} ${PROJECT_LIST} PARENT_SCOPE)
endfunction(list_subdirectories)

function(add_subdirectories)
  cmake_parse_arguments(ARG "" "" "EXCLUDES" ${ARGN})

  list_subdirectories(PROJECT_DIRS EXCLUDES ${ARG_EXCLUDES})

  foreach(PROJECT_DIR IN ITEMS ${PROJECT_DIRS})
    add_subdirectory(${PROJECT_DIR})
  endforeach(PROJECT_DIR)
endfunction(add_subdirectories)
