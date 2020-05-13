function(_import_TensorFlowLite2 sources_dir install_dir)
    # TODO to add includes from third-party libs in more predictable way
    if(NOT TARGET tensorflow-lite2)
        add_library(tensorflow-lite2 INTERFACE)
        target_include_directories(tensorflow-lite2 SYSTEM INTERFACE ${sources_dir})
        target_include_directories(tensorflow-lite2 SYSTEM INTERFACE ${sources_dir}/bazel-out/arm64-v8a-opt/bin/external/flatbuffers/_virtual_includes/flatbuffers)
        target_include_directories(tensorflow-lite2 SYSTEM INTERFACE ${sources_dir}/../absl)
        target_link_libraries(tensorflow-lite2 INTERFACE ${install_dir}/libutils.a)
        target_link_libraries(tensorflow-lite2 INTERFACE ${install_dir}/libtensorflowlite.so)
        target_link_libraries(tensorflow-lite2 INTERFACE log)
        target_link_libraries(tensorflow-lite2 INTERFACE ${install_dir}/libtensorflowlite_gpu_delegate.so)
        target_link_libraries(tensorflow-lite2 INTERFACE EGL GLESv3)
        target_link_libraries(tensorflow-lite2 INTERFACE ${install_dir}/libhexagon_delegate.a)
        target_link_libraries(tensorflow-lite2 INTERFACE ${install_dir}/libhexagon_delegate_kernel.a)
        target_link_libraries(tensorflow-lite2 INTERFACE ${install_dir}/libhexagon_implementation.a)
        target_link_libraries(tensorflow-lite2 INTERFACE ${install_dir}/libhexagon_utils.a)
        target_link_libraries(tensorflow-lite2 INTERFACE ${install_dir}/libop_builder.a)
    endif(NOT TARGET tensorflow-lite2)

    set(TensorFlowLite2_FOUND TRUE PARENT_SCOPE)
endfunction(_import_TensorFlowLite2)

function(_check_maybe_TensorFlowLite2_already_built VAR LIBDIR)
    set(FOUND TRUE)

    if(NOT EXISTS "${LIBDIR}/libtensorflowlite.so")
        set(FOUND FALSE)
    endif()

    if(NOT EXISTS "${LIBDIR}/libtensorflowlite_gpu_delegate.so")
        set(FOUND FALSE)
    endif()

    if(NOT EXISTS "${LIBDIR}/libutils.a")
        set(FOUND FALSE)
    endif()

    if(NOT EXISTS "${LIBDIR}/libhexagon_interface.so")
        set(FOUND FALSE)
    endif()

    if(NOT EXISTS "${LIBDIR}/libhexagon_delegate.a")
        set(FOUND FALSE)
    endif()

    if(NOT EXISTS "${LIBDIR}/libhexagon_delegate_kernel.a")
        set(FOUND FALSE)
    endif()

    if(NOT EXISTS "${LIBDIR}/libhexagon_implementation.a")
        set(FOUND FALSE)
    endif()

    if(NOT EXISTS "${LIBDIR}/libhexagon_utils.a")
        set(FOUND FALSE)
    endif()


    if(NOT EXISTS "${LIBDIR}/libop_builder.a")
        set(FOUND FALSE)
    endif()

    set(${VAR} ${FOUND} PARENT_SCOPE)
endfunction(_check_maybe_TensorFlowLite2_already_built)

function(_check_that_all_preconditions_satisfied flag)
    # TODO to think more correct way to define paths to adnroid ndk and android sdk

    set(all_are_ok TRUE)

    find_program(BAZEL_PATH bazel)
    if(NOT BAZEL_PATH)
        message(WARNING "BAZEL NOT FOUND. Please install bazel 2.0.0 to build TensorFlow lite 2.")
        set(all_are_ok FALSE)
    endif(NOT BAZEL_PATH)

    set(android_ndk_found TRUE)
    if (NOT EXISTS ${CMAKE_SOURCE_DIR}/../../tools/cross/ndk/r20/ndk)
        set(android_ndk_found FALSE)
    endif()

    if (NOT EXISTS ${CMAKE_SOURCE_DIR}/../../tools/cross/ndk/r20/ndk/platforms)
        set(android_ndk_found FALSE)
    endif()

    if (NOT EXISTS ${CMAKE_SOURCE_DIR}/../../tools/cross/ndk/r20/ndk/toolchains)
        set(android_ndk_found FALSE)
    endif()

    if (NOT EXISTS ${CMAKE_SOURCE_DIR}/../../tools/cross/ndk/r20/ndk/sysroot)
        set(android_ndk_found FALSE)
    endif()

    if (NOT EXISTS ${CMAKE_SOURCE_DIR}/../../tools/cross/ndk/r20/ndk/source.properties)
        set(android_ndk_found FALSE)
    endif()

    if(NOT android_ndk_found)
        set(all_are_ok FALSE)
        message(WARNING "Android NDK is not found in directory ${CMAKE_SOURCE_DIR}/../../tools/cross/ndk/r20/ndk. Please install it using command ./tools/cross/install_android_ndk.sh")
    endif()

    set(android_sdk_found TRUE)
    if (NOT EXISTS ${CMAKE_SOURCE_DIR}/../../tools/cross/android_sdk)
        set(android_sdk_found FALSE)
    endif()

    if (NOT EXISTS ${CMAKE_SOURCE_DIR}/../../tools/cross/android_sdk/build-tools)
        set(android_sdk_found FALSE)
    endif()

    if (NOT EXISTS ${CMAKE_SOURCE_DIR}/../../tools/cross/android_sdk/platforms)
        set(android_sdk_found FALSE)
    endif()

    if (NOT EXISTS ${CMAKE_SOURCE_DIR}/../../tools/cross/android_sdk/platform-tools)
        set(android_sdk_found FALSE)
    endif()

    if (NOT EXISTS ${CMAKE_SOURCE_DIR}/../../tools/cross/android_sdk/tools)
        set(android_sdk_found FALSE)
    endif()

    if (NOT android_sdk_found)
        set(all_are_ok FALSE)
        message(WARNING "Android SDK is not found in directory ${CMAKE_SOURCE_DIR}/../../tools/cross/android_sdk. Please install it using script tools/cross/install_android_sdk.sh")
    endif()

    set(${flag} ${all_are_ok} PARENT_SCOPE)
endfunction()

function(_build target_name result_name alias path_to_sources target_prefix path_to_install_dir)
    set(result ${path_to_sources}/bazel-bin/tensorflow/lite${target_prefix}/${result_name})
    if (NOT EXISTS ${result} OR NOT result_name)
        execute_process(
            COMMAND /usr/bin/bazel build --config=android_arm64 --config=monolithic //tensorflow/lite${target_prefix}:${target_name}
            WORKING_DIRECTORY ${path_to_sources})
        return()
    endif()

    if (NOT EXISTS ${path_to_install_dir}/${alias})
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E copy ${result} "${path_to_install_dir}/${alias}"
            WORKING_DIRECTORY ${path_to_sources})
    endif()
endfunction(_build)

function(_build_TensorFlowLite2 sources_dir install_dir)
    _check_maybe_TensorFlowLite2_already_built(already_built ${install_dir})
    if (already_built)
        return()
    endif()

    _check_that_all_preconditions_satisfied(all_are_ok)
    if(NOT all_are_ok)
        return()
    endif()

    # work around. This variable is setting during install NNPACK and
    #that is failing configuring of tensorflow lite2 build system
    set(python_path $ENV{PYTHONPATH})
    unset(ENV{PYTHONPATH})

    # configure tensorflow lite 2 build system
    # TODO to think more correct way to define paths to android ndk and android sdk
    if (NOT EXISTS ${sources_dir}/.tf_configure.bazelrc)
        execute_process(
            COMMAND bash "-c" "printf '\nN\nN\nN\nN\n\ny\n${CMAKE_SOURCE_DIR}/../../tools/cross/ndk/r20/ndk\n21\n${CMAKE_SOURCE_DIR}/../../tools/cross/android_sdk\n28\n28.0.0\n' | PYTHON_BIN_PATH=`which python3` ${sources_dir}/configure"
            WORKING_DIRECTORY ${sources_dir})
    endif()

    # create install dir if need
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${install_dir}" WORKING_DIRECTORY ${sources_dir})

    _build(tensorflowlite libtensorflowlite.so libtensorflowlite.so ${sources_dir} "" ${install_dir})
    _build(libtensorflowlite_gpu_delegate.so libtensorflowlite_gpu_delegate.so libtensorflowlite_gpu_delegate.so ${sources_dir} /delegates/gpu ${install_dir})
    _build(utils libutils.a libutils.a ${sources_dir} /tools/evaluation ${install_dir})
    _build(hexagon_nn_header "" "" ${sources_dir} /experimental/delegates/hexagon/hexagon_nn ${install_dir})
    _build(libhexagon_interface libhexagon_interface.so libhexagon_interface.so ${sources_dir} /experimental/delegates/hexagon/hexagon_nn ${install_dir})
    _build(hexagon_delegate libhexagon_delegate.a libhexagon_delegate.a ${sources_dir} /experimental/delegates/hexagon ${install_dir})
    _build(hexagon_delegate_kernel libhexagon_delegate_kernel.a libhexagon_delegate_kernel.a ${sources_dir} /experimental/delegates/hexagon ${install_dir})
    _build(hexagon_implementation libhexagon_implementation.a libhexagon_implementation.a ${sources_dir} /experimental/delegates/hexagon ${install_dir})
    _build(utils libutils.a libhexagon_utils.a ${sources_dir} /experimental/delegates/hexagon ${install_dir})
    _build(op_builder libop_builder.a libop_builder.a ${sources_dir} /experimental/delegates/hexagon/builders ${install_dir})

    set(ENV{PYTHONPATH} ${python_path})
endfunction(_build_TensorFlowLite2)

nnas_find_package(TensorFlowLite2Source QUIET)
if(NOT TensorFlowLite2Source_FOUND)
    message(WARNING "Tensorflow Lite 2 sources not found")
    return()
endif(NOT TensorFlowLite2Source_FOUND)

set(sources_dir ${TensorFlowLite2Source_DIR})
set(install_dir ${EXT_OVERLAY_DIR}/lib)
if(BUILD_TENSORFLOW_LITE2)
    _build_TensorFlowLite2(${sources_dir} ${install_dir})
endif(BUILD_TENSORFLOW_LITE2)
_import_TensorFlowLite2(${sources_dir} ${install_dir})
