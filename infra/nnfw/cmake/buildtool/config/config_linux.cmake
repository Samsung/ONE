#
# linux common compile options
#

# Remove warning: ignoring attributes on template argument (ACL, Eigen, etc)
# https://github.com/ARM-software/ComputeLibrary/issues/330
set(FLAGS_CXXONLY ${FLAGS_CXXONLY} "-Wno-ignored-attributes")

# Disable annoying ABI compatibility warning.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
  list(APPEND FLAGS_CXXONLY "-Wno-psabi")
endif()
