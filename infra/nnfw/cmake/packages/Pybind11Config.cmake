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
