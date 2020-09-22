if(NOT BUILD_ANDROID_BENCHMARK_APP)
  return()
endif(NOT BUILD_ANDROID_BENCHMARK_APP)

if(NOT ANDROID)
  message(STATUS "Sample app is disabled as non-Android build")
  return()
endif()

if(NOT DEFINED ANDROID_BUILD_TOOLS_DIR)
  message(STATUS "Sample app is disabled as ANDROID_BUILD_TOOLS_DIR is not defined")
  return()
endif()

if(NOT DEFINED ANDROID_SDK_DIR)
  message(STATUS "Sample app is disabled as ANDROID_SDK_DIR is not defined")
  return()
endif()

if(NOT DEFINED TFLITE_MODEL_PATH)
  message(STATUS "Sample app is disabled as TFLITE_MODEL_PATH is not defined")
  return()
endif()

nnfw_find_package(ARMCompute REQUIRED)

if(NOT CORE_LIBRARY OR NOT RUNTIME_LIBRARY)
  message(STATUS "Sample app is disabled as ARM Compute Library is missing")
  return()
endif()

get_filename_component(TFLITE_MODEL_FILENAME ${TFLITE_MODEL_PATH} NAME)

set(ANDROID_API_VERSION 27)
set(ANDROID_PLATFORM_DIR ${ANDROID_SDK_DIR}/platforms/android-${ANDROID_API_VERSION})

set(GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/gen)
set(OBJ_DIR ${CMAKE_CURRENT_BINARY_DIR}/obj)
set(PKG_DIR ${CMAKE_CURRENT_BINARY_DIR}/pkg)
set(LIB_DIR ${PKG_DIR}/lib/arm64-v8a)
set(ASSETS_DIR ${PKG_DIR}/assets)
set(SRC_LIST ${CMAKE_CURRENT_BINARY_DIR}/src.list)

if (ANDROID_BOOST_ROOT)
  set(BOOST_ROOT ${ANDROID_BOOST_ROOT})
endif (ANDROID_BOOST_ROOT)

nnfw_find_package(Boost REQUIRED)

add_library(android_benchmark_native SHARED cpp/ndk_main.cpp)
target_compile_definitions(android_benchmark_native PRIVATE MODEL_NAME="${TFLITE_MODEL_FILENAME}")
target_include_directories(android_benchmark_native PRIVATE ${NNAS_EXTERNALS_DIR}/tensorflow)
target_include_directories(android_benchmark_native PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(android_benchmark_native nnfw_lib_tflite)
target_link_libraries(android_benchmark_native nnfw_lib_misc)
target_link_libraries(android_benchmark_native log)

nnas_find_package(FlatBuffersSource EXACT 1.12 REQUIRED)
target_include_directories(android_benchmark_native PUBLIC ${FlatBuffersSource_DIR}/include .)

add_custom_target(android-benchmark-apk ALL
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${GEN_DIR}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${GEN_DIR}
  COMMAND ${ANDROID_BUILD_TOOLS_DIR}/aapt package -m -J ${GEN_DIR}
                                                  -M ${CMAKE_CURRENT_SOURCE_DIR}/AndroidManifest.xml
                                                  -S ${CMAKE_CURRENT_SOURCE_DIR}/res
                                                  -I ${ANDROID_PLATFORM_DIR}/android.jar
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${OBJ_DIR}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OBJ_DIR}
  COMMAND ${CMAKE_COMMAND} -E remove -f ${SRC_LIST}
  COMMAND find ${CMAKE_CURRENT_SOURCE_DIR}/java -name '*.java' >> ${SRC_LIST}
  COMMAND find ${GEN_DIR} -name '*.java' >> ${SRC_LIST}
  COMMAND javac -d ${OBJ_DIR} -source 1.7 -target 1.7 -bootclasspath "${JAVA_HOME}/jre/lib/rt.jar"
                -classpath ${ANDROID_PLATFORM_DIR}/android.jar @${SRC_LIST}
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${PKG_DIR}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${PKG_DIR}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${ASSETS_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy ${TFLITE_MODEL_PATH} ${ASSETS_DIR}/model.tflite
  COMMAND ${ANDROID_BUILD_TOOLS_DIR}/dx --dex --output=${PKG_DIR}/classes.dex ${OBJ_DIR}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${LIB_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy ${CORE_LIBRARY} ${LIB_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy ${RUNTIME_LIBRARY} ${LIB_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:android_benchmark_native> ${LIB_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:onert> ${LIB_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:onert_backend_acl_cl> ${LIB_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:arm_compute_ex> ${LIB_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:onert_backend_acl_neon> ${LIB_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:onert_backend_cpu> ${LIB_DIR}
  COMMAND ${CMAKE_COMMAND} -E echo ${ANDROID_BUILD_TOOLS_DIR}/aapt package -f -0 tflite -M ${CMAKE_CURRENT_SOURCE_DIR}/AndroidManifest.xml -S ${CMAKE_CURRENT_SOURCE_DIR}/res/ -I ${ANDROID_PLATFORM_DIR}/android.jar -F ${CMAKE_CURRENT_BINARY_DIR}/android-benchmark.unsigned.pkg ${PKG_DIR}
  COMMAND ${ANDROID_BUILD_TOOLS_DIR}/aapt package -f
                                                  -0 tflite
                                                  -M ${CMAKE_CURRENT_SOURCE_DIR}/AndroidManifest.xml
                                                  -S ${CMAKE_CURRENT_SOURCE_DIR}/res/
                                                  -I ${ANDROID_PLATFORM_DIR}/android.jar
                                                  -F ${CMAKE_CURRENT_BINARY_DIR}/android-benchmark.unsigned.pkg
                                                  ${PKG_DIR}
  DEPENDS android_benchmark_native onert onert_backend_acl_cl onert_backend_acl_neon onert_backend_cpu)
