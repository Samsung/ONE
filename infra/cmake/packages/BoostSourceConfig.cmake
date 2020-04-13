function(_BoostSource_import)
  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  # EXTERNAL_DOWNLOAD_SERVER will be overwritten by CI server to use mirror server.
  envoption(EXTERNAL_DOWNLOAD_SERVER "http://sourceforge.net")
  set(BOOST_URL ${EXTERNAL_DOWNLOAD_SERVER}/projects/boost/files/boost/1.58.0/boost_1_58_0.tar.gz)
  ExternalSource_Get(BOOST ${DOWNLOAD_BOOST} ${BOOST_URL})

  set(BoostSource_DIR ${BOOST_SOURCE_DIR} PARENT_SCOPE)
  set(BoostSource_FOUND ${BOOST_SOURCE_GET} PARENT_SCOPE)
endfunction(_BoostSource_import)

_BoostSource_import()
