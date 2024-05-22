function(_XnnpackSource_import)
  if(NOT ${DOWNLOAD_XNNPACK})
    set(XnnpackSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT ${DOWNLOAD_XNNPACK})

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  # xnnpack latest commit (2024.05.20)
  # xnnpack in tflite v2.16.1 is not stable on armv7l gbs and linux cross build process (assembly microkernel build issue)
  # Patch: workaround to resolve build fail by forcing disable using armv8 feature on gbs build and arm linux cross build under gcc 10
  envoption(XNNPACK_URL ${EXTERNAL_DOWNLOAD_SERVER}/google/XNNPACK/archive/fcb36699c67201ceff7358df42730809e8f2c9cc.tar.gz)
  ExternalSource_Download(XNNPACK
    DIRNAME XNNPACK
    URL ${XNNPACK_URL}
    PATCH ${CMAKE_CURRENT_LIST_DIR}/XnnpackSource.patch)

  set(XnnpackSource_DIR ${XNNPACK_SOURCE_DIR} PARENT_SCOPE)
  set(XnnpackSource_FOUND TRUE PARENT_SCOPE)
endfunction(_XnnpackSource_import)

_XnnpackSource_import()
