# Disable warning messages from external source code
if(DISABLE_EXTERNAL_WARNING)
  add_compile_options(-w)
endif(DISABLE_EXTERNAL_WARNING)

# Add all subdirectories.
# Each library in sub-directory must have it's own CMakeLists.txt
# to build library's binaries or to support interface.
add_subdirectories()
