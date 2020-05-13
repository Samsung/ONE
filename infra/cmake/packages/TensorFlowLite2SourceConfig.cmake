function(_TensorFlowLite2Source_import platform)
    if(NOT DOWNLOAD_TENSORFLOW_LITE2)
        set(TensorFlowLite2Source_FOUND FALSE PARENT_SCOPE)
        return()
    endif(NOT DOWNLOAD_TENSORFLOW_LITE2)

    nnas_include(ExternalSourceTools)
    nnas_include(OptionTools)

    envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
    set(TENSORFLOW_LITE2_URL ${EXTERNAL_DOWNLOAD_SERVER}/tensorflow/tensorflow/archive/v2.2.0-rc0.tar.gz)
    ExternalSource_Get("tensorflow2-${platform}" ${DOWNLOAD_TENSORFLOW_LITE2} ${TENSORFLOW_LITE2_URL})


    set(TensorFlowLite2Source_DIR ${tensorflow2-${platform}_SOURCE_DIR} PARENT_SCOPE)
    set(TensorFlowLite2Source_FOUND ${tensorflow2-${platform}_SOURCE_GET} PARENT_SCOPE)
endfunction(_TensorFlowLite2Source_import)


if (${TARGET_OS} STREQUAL "linux" AND ${TARGET_ARCH} STREQUAL "x86_64")
    set(platform ${TARGET_ARCH}.${TARGET_OS})
elseif (${TARGET_OS} STREQUAL "android" AND ${TARGET_ARCH} STREQUAL "aarch64")
    set(platform ${TARGET_ARCH}.${TARGET_OS})
else()
    message("TensorFlow Lite 2 does not support platform ${TARGET_ARCH}.${TARGET_OS} yet. Build will be skipped.")
    return()
endif()

_TensorFlowLite2Source_import(${platform})
