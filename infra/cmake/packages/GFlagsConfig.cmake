function(_GFlags_import)
  if(TARGET gflags)
    set(GFlags_FOUND
        True
        PARENT_SCOPE)
    return()
  endif()

  nnas_find_package(GFlagsSource QUIET)

  if(GFlagsSource_FOUND)
    nnas_include(ExternalProjectTools)
    # build shared multi-threading gflag library
    set(BUILD_SHARED_LIBS On)
    set(BUILD_STATIC_LIBS Off)
    set(BUILD_gflags_LIB On)
    set(BUILD_gflags_nothreads_LIB Off)
    add_extdirectory(${GFLAGS_SOURCE_DIR} gflags)
  else(GFlagsSource_FOUND)
    set(GFLAGS_ROOT_DIR
        ""
        CACHE PATH "Folder contains GFlags")
    find_path(GFLAGS_INCLUDE_DIR gflags/gflags.h PATHS ${GFLAGS_ROOT_DIR})
    find_library(GFLAGS_LIBRARY gflags)

    if(NOT GFLAGS_INCLUDE_DIR)
      set(GFlags_FOUND
          False
          PARENT_SCOPE)
      return()
    endif(NOT GFLAGS_INCLUDE_DIR)

    add_library(gflags INTERFACE)
    target_include_directories(gflags INTERFACE ${GFLAGS_INCLUDE_DIR})
    target_link_libraries(gflags INTERFACE ${GFLAGS_LIBRARY})
  endif(GFlagsSource_FOUND)

  set(GFlags_FOUND
      True
      PARENT_SCOPE)
endfunction(_GFlags_import)

_GFlags_import()
