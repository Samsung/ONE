function(_PeachpySource_import)
    nnas_include(ExternalSourceTools)
    nnas_include(OptionTools)

    envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
    set(PEACHPY_URL ${EXTERNAL_DOWNLOAD_SERVER}/Maratyszcza/PeachPy/archive/01d15157a973a4ae16b8046313ddab371ea582db.tar.gz)

    ExternalSource_Get("PYTHON_PEACHPY" ${DOWNLOAD_NNPACK} ${PEACHPY_URL})
    FIND_PACKAGE(PythonInterp)

    nnfw_find_package(SixSource REQUIRED)
    nnfw_find_package(Enum34Source REQUIRED)
    nnfw_find_package(OpcodesSource REQUIRED)

    # Generate opcodes:
    SET(ENV{PYTHONPATH} ${PYTHON_PEACHPY_SOURCE_DIR}:${PYTHON_SIX_SOURCE_DIR}:${PYTHON_ENUM_SOURCE_DIR}:${PYTHON_OPCODES_SOURCE_DIR})
    EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} ./codegen/x86_64.py
            WORKING_DIRECTORY ${PYTHON_PEACHPY_SOURCE_DIR}
            RESULT_VARIABLE BUILT_PP)

    if(NOT BUILT_PP EQUAL 0)
			# Mark PYTHON_PEACHPY_SOURCE_FOUND as FALSE if source generation fails
      set(PYTHON_PEACHPY_SOURCE_FOUND FALSE PARENT_SCOPE)
      return()
    endif(NOT BUILT_PP EQUAL 0)

    set(PYTHON_PEACHPY_SOURCE_DIR ${PYTHON_PEACHPY_SOURCE_DIR} PARENT_SCOPE)
    set(PYTHON_PEACHPY_SOURCE_FOUND ${PYTHON_PEACHPY_SOURCE_GET} PARENT_SCOPE)
endfunction(_PeachpySource_import)

_PeachpySource_import()
