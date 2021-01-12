# Copyright (c) 2021 ARM Limited. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.19.0)
set(LUCI_MICRO)
set(MBED_PATH ${CMAKE_CURRENT_SOURCE_DIR}/mbed-os CACHE INTERNAL "")
set(MBED_CONFIG_PATH ${CMAKE_CURRENT_SOURCE_DIR}/.mbedbuild CACHE INTERNAL "")
set(APP_TARGET luci-micro)
set(LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib CACHE INTERNAL "")


include(${MBED_PATH}/tools/cmake/app.cmake)
SET(GCC_COVERAGE_COMPILE_FLAGS "-fexceptions -DLUCI_MICRO")
SET(GCC_COVERAGE_LINK_FLAGS    "")

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})

add_subdirectory(${LIB_PATH}/stdex)
add_subdirectory(${LIB_PATH}/oops)
add_subdirectory(${LIB_PATH}/pepper-str)
add_subdirectory(${LIB_PATH}/logo)
add_subdirectory(${LIB_PATH}/logo-core)
add_subdirectory(${LIB_PATH}/loco)
add_subdirectory(${LIB_PATH}/locomotiv)
add_subdirectory(${LIB_PATH}/angkor)

add_subdirectory(${MBED_PATH})

add_executable(${APP_TARGET}
    main.cpp
)

mbed_configure_app_target(${APP_TARGET})

mbed_set_mbed_target_linker_script(${APP_TARGET})

project(${APP_TARGET})

target_link_libraries(${APP_TARGET} mbed-os)
target_link_libraries(${APP_TARGET} stdex)
target_link_libraries(${APP_TARGET} oops)
target_link_libraries(${APP_TARGET} pepper_str)
target_link_libraries(${APP_TARGET} logo)
target_link_libraries(${APP_TARGET} logo_core)
target_link_libraries(${APP_TARGET} loco)
target_link_libraries(${APP_TARGET} locomotiv)
target_link_libraries(${APP_TARGET} angkor)

mbed_set_post_build(${APP_TARGET})

option(VERBOSE_BUILD "Have a verbose build process")
if(VERBOSE_BUILD)
    set(CMAKE_VERBOSE_MAKEFILE ON)
endif()