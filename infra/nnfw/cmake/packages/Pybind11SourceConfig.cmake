function(_Pybind11Source_import)
  if(NOT DOWNLOAD_PYBIND11)
    set(Pybind11Source_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_PYBIND11)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  envoption(PYBIND11_URL ${EXTERNAL_DOWNLOAD_SERVER}/pybind/pybind11/archive/v2.11.1.tar.gz)

  ExternalSource_Download(PYBIND11 ${PYBIND11_URL})

  set(Pybind11Source_DIR ${PYBIND11_SOURCE_DIR} PARENT_SCOPE)
  set(Pybind11Source_FOUND TRUE PARENT_SCOPE)
endfunction(_Pybind11Source_import)

_Pybind11Source_import()
