message(STATUS "ONERT MICRO CORE BUILD BEGIN")

set(SOURCES
        OMKernelType.cpp
        OMRuntimeContext.cpp
        OMRuntimeStorage.cpp
        OMRuntimeGraph.cpp
        OMRuntimeModule.cpp
        OMUtils.cpp
        OMDataType.cpp
        memory/OMMemoryManager.cpp
        memory/OMRuntimeAllocator.cpp
        reader/OMCircleReader.cpp
        reader/OMWeightOnlyFormatReader.cpp)

add_library(${OM_CORE_LIB} STATIC ${SOURCES})

target_include_directories(${OM_CORE_LIB} PUBLIC "${OM_INCLUDE_DIR}")
target_include_directories(${OM_CORE_LIB} PUBLIC "${GENERATED_INCLUDE_DIR}")

message(STATUS "ONERT MICRO CORE BUILD FINISHED")
