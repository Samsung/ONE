function(_NoniusSource_import)
  if(NOT ${DOWNLOAD_NONIUS})
    set(NoniusSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT ${DOWNLOAD_NONIUS})

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  set(NONIUS_URL ${EXTERNAL_DOWNLOAD_SERVER}/libnonius/nonius/archive/v1.2.0-beta.1.tar.gz)
  ExternalSource_Download("NONIUS" ${NONIUS_URL})

  if(BUILD_KBENCHMARK)
    # Copy html_report_template.g.h++ file to externals/nonius.
    # This header file is modified to show the html summary view according to the layer in kbenchmark.
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy
                    "${CMAKE_CURRENT_LIST_DIR}/Nonius/html_report_template.g.h++"
                    "${Nonius_Source_DIR}/include/nonius/detail")
  endif(BUILD_KBENCHMARK)

  set(NoniusSource_DIR ${NONIUS_SOURCE_DIR} PARENT_SCOPE)
  set(NoniusSource_FOUND TRUE PARENT_SCOPE)
endfunction(_NoniusSource_import)

_NoniusSource_import()
