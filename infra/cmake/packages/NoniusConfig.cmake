function(_Nonius_import)
  nnas_find_package(NoniusSource QUIET)

  if(NOT NoniusSource_FOUND)
    set(Nonius_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT NoniusSource_FOUND)

  if(NOT TARGET nonius)
    message(STATUS "Found nonius: TRUE")
    add_library(nonius INTERFACE)
    target_include_directories(nonius INTERFACE "${NoniusSource_DIR}/include")
  endif(NOT TARGET nonius)

  if(BUILD_KBENCHMARK)
    # Copy html_report_template.g.h++ file to externals/nonius.
    # This header file is modified to show the html summary view according to the layer in kbenchmark.
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy
                    "${CMAKE_CURRENT_LIST_DIR}/Nonius/html_report_template.g.h++"
                    "${NoniusSource_DIR}/include/nonius/detail")
  endif(BUILD_KBENCHMARK)

  set(Nonius_FOUND TRUE PARENT_SCOPE)
endfunction(_Nonius_import)

_Nonius_import()
