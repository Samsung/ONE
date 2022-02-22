macro(initialize_pal)
    nnas_find_package(TensorFlowSource EXACT 2.8.0 QUIET)
    nnas_find_package(TensorFlowGEMMLowpSource EXACT 2.8.0 QUIET)
    nnas_find_package(TensorFlowEigenSource EXACT 2.8.0 QUIET)
    nnas_find_package(TensorFlowRuySource EXACT 2.8.0 QUIET)

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

    find_package(Threads REQUIRED)

    set(PAL_INITIALIZED TRUE)
endmacro()

macro(add_pal_to_target TGT)
    target_include_directories(${TGT} PRIVATE "${PAL}")
    target_include_directories(${TGT} SYSTEM PRIVATE
            "${TensorFlowRuySource_DIR}"
            "${TensorFlowGEMMLowpSource_DIR}"
            "${TensorFlowEigenSource_DIR}"
            "${TensorFlowSource_DIR}")
    target_include_directories(${TGT} PRIVATE ${LUCI_INTERPRETER_PAL_DIR})

    # TODO put it back, I changed my mind.
    # instead add sources with visitors in this library
    set(PAL_SOURCES ${TensorFlowSource_DIR}/tensorflow/lite/kernels/internal/tensor_utils.cc
            ${TensorFlowSource_DIR}/tensorflow/lite/kernels/internal/reference/portable_tensor_utils.cc
            ${TensorFlowSource_DIR}/tensorflow/lite/kernels/internal/quantization_util.cc)

    if(BUILD_ARM32_NEON)
        # NOTE may need to revise this list for version upgrade
        set(PAL_SOURCES ${PAL_SOURCES}
                ${TensorFlowSource_DIR}/tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.cc
                ${TensorFlowSource_DIR}/tensorflow/lite/kernels/internal/optimized/cpu_check.cc
                ${TensorFlowRuySource_DIR}/ruy/allocator.cc
                ${TensorFlowRuySource_DIR}/ruy/block_map.cc
                ${TensorFlowRuySource_DIR}/ruy/blocking_counter.cc
                ${TensorFlowRuySource_DIR}/ruy/context_get_ctx.cc
                ${TensorFlowRuySource_DIR}/ruy/cpuinfo.cc
                ${TensorFlowRuySource_DIR}/ruy/ctx.cc
                ${TensorFlowRuySource_DIR}/ruy/denormal.cc
                ${TensorFlowRuySource_DIR}/ruy/frontend.cc
                ${TensorFlowRuySource_DIR}/ruy/pack_arm.cc
                ${TensorFlowRuySource_DIR}/ruy/prepacked_cache.cc
                ${TensorFlowRuySource_DIR}/ruy/prepare_packed_matrices.cc
                ${TensorFlowRuySource_DIR}/ruy/system_aligned_alloc.cc
                ${TensorFlowRuySource_DIR}/ruy/thread_pool.cc
                ${TensorFlowRuySource_DIR}/ruy/trmul.cc
                ${TensorFlowRuySource_DIR}/ruy/tune.cc
                ${TensorFlowRuySource_DIR}/ruy/wait.cc
                ${TensorFlowRuySource_DIR}/ruy/kernel_arm32.cc
                )
    endif(BUILD_ARM32_NEON)

    add_library(luci_interpreter_linux_pal STATIC ${PAL_SOURCES})
    target_compile_options(luci_interpreter_linux_pal PUBLIC "-mfpu=vfp") # to disable __ARM_NEON
    set_target_properties(luci_interpreter_linux_pal PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(luci_interpreter_linux_pal SYSTEM PRIVATE
            "${TensorFlowRuySource_DIR}"
            "${TensorFlowGEMMLowpSource_DIR}"
            "${TensorFlowEigenSource_DIR}"
            "${TensorFlowSource_DIR}"
    )

    target_link_libraries(${TGT} PRIVATE Threads::Threads luci_interpreter_linux_pal)
endmacro()
