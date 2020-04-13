function(_PeachpySource_import)
    nnas_include(ExternalSourceTools)
    nnas_include(OptionTools)

    envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
    set(PEACHPY_URL ${EXTERNAL_DOWNLOAD_SERVER}/Maratyszcza/Opcodes/archive/6e2b0cd9f1403ecaf164dea7019dd54db5aea252.tar.gz)
    ExternalSource_Get("PYTHON_OPCODES" ${DOWNLOAD_NNPACK} ${PEACHPY_URL})

    set(PYTHON_OPCODES_SOURCE_DIR ${PYTHON_OPCODES_SOURCE_DIR} PARENT_SCOPE)
    set(PYTHON_OPCODES_SOURCE_FOUND ${PYTHON_OPCODES_SOURCE_GET} PARENT_SCOPE)
endfunction(_PeachpySource_import)

_PeachpySource_import()
