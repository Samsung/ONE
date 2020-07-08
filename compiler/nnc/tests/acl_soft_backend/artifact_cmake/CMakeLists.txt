cmake_minimum_required(VERSION 3.5)
project(nnc_test)

set(CMAKE_CXX_STANDARD 11)

set(ODROID_MIRROR_DIR $ENV{ODROID_MIRROR_DIR})
set(ODROID_ACL_DIR $ENV{ODROID_ACL_DIR})
set(ODROID_ACL_INC_DIR ${ODROID_ACL_DIR}/include)
set(ODROID_ACL_BUILD_DIR ${ODROID_ACL_DIR}/build)

find_library(OPEN_CL OpenCL /usr/lib/arm-linux-gnueabihf)
find_library(ARM_COMPUTE arm_compute PATHS ${ODROID_ACL_BUILD_DIR})
find_library(ARM_COMPUTE_CORE arm_compute_core PATHS ${ODROID_ACL_BUILD_DIR})
set(HDF5_USE_STATIC_LIBRARIES OFF)
unset(HDF5_CXX_LIBRARY_hdf5 CACHE)
unset(HDF5_CXX_LIBRARY_hdf5_cpp CACHE)
find_package(HDF5 COMPONENTS CXX REQUIRED)

add_executable(nnc_test main.cpp AclArtifact.cpp)

target_include_directories(nnc_test PRIVATE . ${ODROID_MIRROR_DIR}${ODROID_ACL_INC_DIR}
    ${ODROID_MIRROR_DIR}${ODROID_ACL_DIR} ${ODROID_MIRROR_DIR}${HDF5_INCLUDE_DIRS})
target_link_libraries(nnc_test ${ARM_COMPUTE} ${ARM_COMPUTE_CORE} ${OPEN_CL} ${HDF5_CXX_LIBRARIES})
target_compile_definitions(nnc_test PRIVATE ARM_COMPUTE_CL)
