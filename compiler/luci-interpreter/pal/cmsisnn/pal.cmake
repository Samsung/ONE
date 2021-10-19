macro(initialize_pal)
    nnas_find_package(TensorFlowSource EXACT 2.6.0 QUIET)
    nnas_find_package(TensorFlowGEMMLowpSource EXACT 2.6.0 QUIET)
    nnas_find_package(TensorFlowEigenSource EXACT 2.6.0 QUIET)
    nnas_find_package(TensorFlowRuySource EXACT 2.6.0 QUIET)
    nnas_find_package(CMSISSource EXACT 5.8.0 QUIET)

    if (NOT TensorFlowSource_FOUND)
        message(STATUS "Skipping luci-interpreter: TensorFlow not found")
        return()
    endif ()

    if (NOT TensorFlowGEMMLowpSource_FOUND)
        message(STATUS "Skipping luci-interpreter: gemmlowp not found")
        return()
    endif ()

    if (NOT TensorFlowEigenSource_FOUND)
        message(STATUS "Skipping luci-interpreter: Eigen not found")
        return()
    endif ()

    if (NOT TensorFlowRuySource_FOUND)
        message(STATUS "Skipping luci-interpreter: Ruy not found")
        return()
    endif ()

    if (NOT CMSISSource_FOUND)
        message(STATUS "Skipping luci-interpreter: CMSISSource not found")
        return()
    endif ()

    set(PAL_INITIALIZED TRUE)
endmacro()

macro(add_pal_to_target TGT)
    target_include_directories(${TGT} PRIVATE "${PAL}")
    target_include_directories(${TGT} PRIVATE
            "${TensorFlowRuySource_DIR}"
            "${TensorFlowGEMMLowpSource_DIR}"
            "${TensorFlowEigenSource_DIR}"
            "${TensorFlowSource_DIR}")
    target_include_directories(${TGT} PRIVATE ${LUCI_INTERPRETER_PAL_DIR})

    set(PAL_SOURCES ${TensorFlowSource_DIR}/tensorflow/lite/kernels/internal/quantization_util.cc)
    add_library(luci_interpreter_cmsisnn_pal STATIC ${PAL_SOURCES})
    set_target_properties(luci_interpreter_cmsisnn_pal PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(luci_interpreter_cmsisnn_pal PRIVATE
            "${TensorFlowRuySource_DIR}"
            "${TensorFlowGEMMLowpSource_DIR}"
            "${TensorFlowEigenSource_DIR}"
            "${TensorFlowSource_DIR}"
    )

    add_subdirectory(${CMSISSource_DIR}/CMSIS/NN ${CMAKE_CURRENT_BINARY_DIR}/CMSISNN)
    target_include_directories(luci_interpreter_cmsisnn_pal PRIVATE
            "${CMSISSource_DIR}/CMSIS/NN/Include"
            "${CMSISSource_DIR}/CMSIS/DSP/Include"
            "${CMSISSource_DIR}/CMSIS/Core/Include")

    target_link_libraries(${TGT} PRIVATE luci_interpreter_cmsisnn_pal)
endmacro()
