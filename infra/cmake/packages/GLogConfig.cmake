function(_GLog_import)
  if(TARGET glog)
    set(GLog_FOUND
        True
        PARENT_SCOPE)
    return()
  endif()

  set(GLOG_ROOT_DIR
      ""
      CACHE PATH "Folder contains Google Log")
  find_path(GLOG_INCLUDE_DIR glog/logging.h PATHS ${GLOG_ROOT_DIR})
  find_library(GLOG_LIBRARY glog)

  if(NOT GLOG_INCLUDE_DIR)
    set(GLog_FOUND
        False
        PARENT_SCOPE)
    return()
  endif(NOT GLOG_INCLUDE_DIR)

  add_library(glog INTERFACE)
  target_include_directories(glog INTERFACE ${GLOG_INCLUDE_DIR} ${GFLAGS_INCLUDE_DIR})
  target_link_libraries(glog INTERFACE ${GLOG_LIBRARY} gflags)

  message(STATUS "Found GLog: TRUE")
  set(GLog_FOUND
      True
      PARENT_SCOPE)
endfunction(_GLog_import)

_GLog_import()
