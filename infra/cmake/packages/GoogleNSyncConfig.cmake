# https://github.com/google/nsync
set(GOOGLE_NSYNC_PREFIX "/usr" CACHE PATH "Where to find Google NSync library")

function(_GoogleNSync_import)
  # Find the header & lib
  find_library(GoogleNSync_C_LIB
    NAMES nsync
    PATHS "${GOOGLE_NSYNC_PREFIX}/lib"
  )

  find_library(GoogleNSync_CPP_LIB
    NAMES nsync_cpp
    PATHS "${GOOGLE_NSYNC_PREFIX}/lib"
  )

  find_path(GoogleNSync_INCLUDE_DIR
    NAMES nsync.h
    PATHS "${GOOGLE_NSYNC_PREFIX}/include"
  )

  message(STATUS "GoogleNSync_C_LIB: ${GoogleNSync_C_LIB}")
  message(STATUS "GoogleNSync_CPP_LIB: ${GoogleNSync_CPP_LIB}")
  message(STATUS "GoogleNSync_INCLUDE_DIR: ${GoogleNSync_INCLUDE_DIR}")

  set(GoogleNSync_FOUND TRUE)

  if(NOT GoogleNSync_C_LIB)
    set(GoogleNSync_FOUND FALSE)
  endif(NOT GoogleNSync_C_LIB)

  if(NOT GoogleNSync_CPP_LIB)
    set(GoogleNSync_FOUND FALSE)
  endif(NOT GoogleNSync_CPP_LIB)

  if(NOT GoogleNSync_INCLUDE_DIR)
    set(GoogleNSync_FOUND FALSE)
  endif(NOT GoogleNSync_INCLUDE_DIR)

  unset(MESSAGE)
  list(APPEND MESSAGE "Found Google NSync")

  if(NOT GoogleNSync_FOUND)
    list(APPEND MESSAGE ": FALSE")
  else(NOT GoogleNSync_FOUND)
    list(APPEND MESSAGE " (include: ${GoogleNSync_INCLUDE_DIR} library: ${GoogleNSync_C_LIB} ${GoogleNSync_CPP_LIB})")

    # Add target
    if(NOT TARGET google_nsync)
      # NOTE IMPORTED target may be more appropriate for this case
      add_library(google_nsync INTERFACE)
      target_link_libraries(google_nsync INTERFACE ${GoogleNSync_C_LIB} ${GoogleNSync_CPP_LIB})
      target_include_directories(google_nsync INTERFACE ${GoogleNSync_INCLUDE_DIR})

      add_library(Google::NSync ALIAS google_nsync)
    endif(NOT TARGET google_nsync)
  endif(NOT GoogleNSync_FOUND)

  message(STATUS ${MESSAGE})
  set(GoogleNSync_FOUND ${GoogleNSync_FOUND} PARENT_SCOPE)
endfunction(_GoogleNSync_import)

_GoogleNSync_import()
