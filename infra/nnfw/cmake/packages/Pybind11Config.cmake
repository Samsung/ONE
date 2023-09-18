# nnfw/Pybind11Config.cmake :

if(${DOWNLOAD_PYBIND11})
  nnas_find_package(Pybind11Source QUIET)
  
  
  if(NOT Pybind11Source_FOUND) # x
    set(Pybind11_FOUND FALSE)
    return()
  endif(NOT Pybind11Source_FOUND)
  
  if(NOT TARGET pybind11) # o
    nnas_include(ExternalBuildTools)
    add_extdirectory(${Pybind11Source_DIR} pybind11 EXCLUDE_FROM_ALL)
  endif(NOT TARGET pybind11)
    
  set(Pybind11_FOUND TRUE)
  return()
endif(${DOWNLOAD_PYBIND11})
### Find and use pre-installed Pybind11
# if(NOT Pybind11_FOUND)
#   # Reset package config directory cache to prevent recursive find
#   unset(Pybind11_DIR CACHE)
#   find_package(Pybind11)
# endif(NOT Pybind11_FOUND)

# if(${Pybind11_FOUND})
#   if(NOT TARGET pybind11)
#     add_library(pybind11 INTERFACE)
#     target_include_directories(pybind11 INTERFACE ${Pybind11_INCLUDE_DIRS})
#   endif(NOT TARGET pybind11)

#   # Additional setup or libraries can be added if necessary

#   set(Pybind11_FOUND TRUE)
# else(${Pybind11_FOUND})
#   find_path(Pybind11_INCLUDE_DIR NAMES pybind11.h)
#   if(Pybind11_INCLUDE_DIR)
#     if(NOT TARGET pybind11)
#       add_library(pybind11 INTERFACE)
#       target_include_directories(pybind11 INTERFACE ${Pybind11_INCLUDE_DIRS})
#     endif(NOT TARGET pybind11)

#     set(Pybind11_FOUND TRUE)
#   endif(Pybind11_INCLUDE_DIR)
# endif(${Pybind11_FOUND})