function(_NNPACK_Import)
    nnas_find_package(NNPACKSource QUIET)

    if(NOT NNPACK_SOURCE_FOUND)
        set(NNPACK_FOUND FALSE PARENT_SCOPE)
        message(STATUS "NNPACK not found")
        return()
    endif(NOT NNPACK_SOURCE_FOUND)

    nnas_find_package(CpuinfoSource REQUIRED)
    nnas_find_package(FP16Source REQUIRED)
    nnas_find_package(FXdivSource REQUIRED)
    nnas_find_package(PSIMDSource REQUIRED)
    nnas_find_package(PthreadpoolSource REQUIRED)
    nnas_find_package(SixSource REQUIRED)
    nnas_find_package(Enum34Source REQUIRED)
    nnas_find_package(OpcodesSource REQUIRED)
    nnas_find_package(PeachpySource QUIET)

    if(NOT PYTHON_PEACHPY_SOURCE_FOUND)
        set(NNPACK_FOUND FALSE PARENT_SCOPE)
        return()
    endif(NOT PYTHON_PEACHPY_SOURCE_FOUND)

    # workaround for CI
    set(THREADS_PTHREAD_ARG "2" CACHE STRING "Forcibly set by CMakeLists.txt." FORCE)
    if(NOT TARGET nnpack)
        # Allows us to build nnpack at build time
        set(NNPACK_BUILD_TESTS OFF CACHE BOOL "")
        set(NNPACK_BUILD_BENCHMARKS OFF CACHE BOOL "")
        set(NNPACK_LIBRARY_TYPE "static" CACHE STRING "")
        set(PTHREADPOOL_LIBRARY_TYPE "static" CACHE STRING "")
        set(CPUINFO_LIBRARY_TYPE "static" CACHE STRING "")
        nnas_include(ExternalProjectTools)
        add_extdirectory("${NNPACK_SOURCE_DIR}" nnpack EXCLUDE_FROM_ALL)
        # We build static versions of nnpack and pthreadpool but link
        # them into a shared library (high-perf-backend), so they need PIC.
        set_property(TARGET nnpack PROPERTY POSITION_INDEPENDENT_CODE ON)
        set_property(TARGET pthreadpool PROPERTY POSITION_INDEPENDENT_CODE ON)
        set_property(TARGET cpuinfo PROPERTY POSITION_INDEPENDENT_CODE ON)
    endif()

    set(NNPACK_FOUND TRUE PARENT_SCOPE)
    set(NNPACK_INCLUDE_DIRS
            $<TARGET_PROPERTY:nnpack,INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:pthreadpool,INCLUDE_DIRECTORIES> PARENT_SCOPE)
    set(NNPACK_LIBRARIES $<TARGET_FILE:nnpack> $<TARGET_FILE:cpuinfo> PARENT_SCOPE)

endfunction(_NNPACK_Import)

_NNPACK_Import()
