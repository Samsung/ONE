# set target platform to run
if(NOT TARGET_ARCH OR "${TARGET_ARCH}" STREQUAL "")
  string(TOLOWER ${CMAKE_SYSTEM_PROCESSOR} TARGET_ARCH)
else()
  string(TOLOWER ${TARGET_ARCH} TARGET_ARCH)
endif()

# If TARGET_ARCH is arm64 change ARCH name to aarch64
if("${TARGET_ARCH}" STREQUAL "arm64")
  set(TARGET_ARCH "aarch64")
endif()

if("${TARGET_ARCH}" STREQUAL "armv8-m")
  set(TARGET_ARCH_BASE "arm")
elseif("${TARGET_ARCH}" STREQUAL "armv7-r")
  set(TARGET_ARCH_BASE "arm")
elseif("${TARGET_ARCH}" STREQUAL "armv7em")
  set(TARGET_ARCH_BASE "arm")
elseif("${TARGET_ARCH}" STREQUAL "armv7l")
  set(TARGET_ARCH_BASE "arm")
elseif("${TARGET_ARCH}" STREQUAL "armv7hl")
  set(TARGET_ARCH_BASE "arm")
elseif("${TARGET_ARCH}" STREQUAL "aarch64")
  set(TARGET_ARCH_BASE "aarch64")
endif()

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
            ${TensorFlowSource_DIR}/tensorflow/lite/kernels/internal/quantization_util.cc
            ${TensorFlowSource_DIR}/tensorflow/lite/kernels/kernel_util.cc
            ${TensorFlowSource_DIR}/tensorflow/lite/c/common.c)

    if(TARGET_ARCH_BASE STREQUAL "arm")
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
    endif(TARGET_ARCH_BASE STREQUAL "arm")

    if(TARGET_ARCH_BASE STREQUAL "aarch64")
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
                ${TensorFlowRuySource_DIR}/ruy/kernel_arm64.cc
                )
    endif(TARGET_ARCH_BASE STREQUAL "aarch64")

    add_library(luci_interpreter_linux_pal STATIC ${PAL_SOURCES})
    set_target_properties(luci_interpreter_linux_pal PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(luci_interpreter_linux_pal SYSTEM PRIVATE
            "${TensorFlowRuySource_DIR}"
            "${TensorFlowGEMMLowpSource_DIR}"
            "${TensorFlowEigenSource_DIR}"
            "${TensorFlowSource_DIR}"
    )

    target_link_libraries(${TGT} PRIVATE Threads::Threads luci_interpreter_linux_pal)
endmacro()
