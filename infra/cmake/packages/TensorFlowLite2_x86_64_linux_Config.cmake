function(_import_TensorFlowLite2 sources_dir install_dir)
    # TODO to think how find includes of tensorflow lite 2 third-parties more correct
    if(NOT TARGET tensorflow-lite2)
        add_library(tensorflow-lite2 INTERFACE)
        target_include_directories(tensorflow-lite2 SYSTEM INTERFACE ${sources_dir})
        target_include_directories(tensorflow-lite2 SYSTEM INTERFACE ${sources_dir}/bazel-tensorflow2-x86_64.linux/external/flatbuffers/include)
        target_include_directories(tensorflow-lite2 SYSTEM INTERFACE ${sources_dir}/bazel-tensorflow2-x86_64.linux/external/com_google_absl)
        target_link_libraries(tensorflow-lite2 INTERFACE ${install_dir}/libtensorflowlite.so)
        target_link_libraries(tensorflow-lite2 INTERFACE ${install_dir}/libtensorflowlite_gpu_delegate.so)
        target_link_libraries(tensorflow-lite2 INTERFACE ${install_dir}/libutils.a)
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

    set(${VAR} ${FOUND} PARENT_SCOPE)
endfunction(_check_maybe_TensorFlowLite2_already_built)

function(_check_that_all_preconditions_satisfied flag)
    set(all_are_ok TRUE)

    find_program(BAZEL_PATH bazel)
    if(NOT BAZEL_PATH)
        message(WARNING "BAZEL NOT FOUND. Please install bazel 2.0.0 to build TensorFlow lite 2.")
        set(all_are_ok FALSE)
    endif(NOT BAZEL_PATH)

    set(${flag} ${all_are_ok} PARENT_SCOPE)
endfunction()

function(_build target_name result_name alias path_to_sources target_prefix path_to_install_dir)
    set(result ${path_to_sources}/bazel-bin/tensorflow/lite${target_prefix}/${result_name})
    if (NOT EXISTS ${result})
        execute_process(
                COMMAND /usr/bin/bazel build --copt -DMESA_EGL_NO_X11_HEADERS --config=monolithic //tensorflow/lite${target_prefix}:${target_name}
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
    if (NOT EXISTS ${sources_dir}/.tf_configure.bazelrc)
    execute_process(COMMAND bash "-c" "printf '\nN\nN\nN\nN\n\nN\n' | PYTHON_BIN_PATH=`which python3` ${sources_dir}/configure"
        WORKING_DIRECTORY ${sources_dir})
    endif()

    # create install dir if need
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${install_dir}" WORKING_DIRECTORY ${sources_dir})

    _build(tensorflowlite libtensorflowlite.so libtensorflowlite.so ${sources_dir} "" ${install_dir})
    _build(libtensorflowlite_gpu_delegate.so libtensorflowlite_gpu_delegate.so libtensorflowlite_gpu_delegate.so ${sources_dir} /delegates/gpu ${install_dir})
    _build(utils libutils.a libutils.a ${sources_dir} /tools/evaluation ${install_dir})

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
