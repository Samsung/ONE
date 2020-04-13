#
# linux common compile options
#

# remove warning from arm cl
# https://github.com/ARM-software/ComputeLibrary/issues/330
set(GCC_VERSION_DISABLE_WARNING 6.0)
if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER GCC_VERSION_DISABLE_WARNING)
  message(STATUS "GCC version higher than ${GCC_VERSION_DISABLE_WARNING}")
  set(FLAGS_CXXONLY ${FLAGS_CXXONLY}
      "-Wno-ignored-attributes"
      )
endif()

# Disable annoying ABI compatibility warning.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
  list(APPEND FLAGS_CXXONLY "-Wno-psabi")
endif()

# lib pthread as a variable (pthread must be disabled on android)
set(LIB_PTHREAD pthread)
