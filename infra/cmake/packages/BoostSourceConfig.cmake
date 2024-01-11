function(_BoostSource_import)
  if(NOT ${DOWNLOAD_BOOST})
    set(BoostSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT ${DOWNLOAD_BOOST})

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  # EXTERNAL_DOWNLOAD_SERVER will be overwritten by CI server to use mirror server.
  envoption(EXTERNAL_DOWNLOAD_SERVER "http://sourceforge.net")
  envoption(BOOST_URL ${EXTERNAL_DOWNLOAD_SERVER}/projects/boost/files/boost/1.84.0/boost_1_84_0.tar.gz)
  ExternalSource_Download(BOOST ${BOOST_URL})

  set(BoostSource_DIR ${BOOST_SOURCE_DIR} PARENT_SCOPE)
  set(BoostSource_FOUND TRUE PARENT_SCOPE)
endfunction(_BoostSource_import)

_BoostSource_import()
