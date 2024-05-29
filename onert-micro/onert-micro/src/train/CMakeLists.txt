message(STATUS "ONERT MICRO TRAIN BUILD BEGIN")

include("${OM_PAL_DIR}/pal.cmake")

initialize_pal()

if (NOT PAL_INITIALIZED)
    message(STATUS "ERROR: PAL Failed to initialize, skip BUILD EXECUTE")
    return()
endif()

set(SOURCES
        OMBackpropExecute.cpp
        OMBackpropExecutionBuilder.cpp
        losses_functions/MSE.cpp
        losses_functions/CrossEntropy.cpp
        metrics/CrossEntropy.cpp
        metrics/MAE.cpp
        metrics/MSE.cpp
        metrics/Accuracy.cpp
        train_optimizers/SGD.cpp
        train_optimizers/Adam.cpp
        )

# Add configure kernels
macro(REGISTER_TRAIN_KERNEL OPERATOR, NODE)
    list(APPEND SOURCES "kernels/${NODE}.cpp")
endmacro(REGISTER_TRAIN_KERNEL)

# To add REGISTER_KERNEL list
include(${KERNEL_TRAIN_REGISTER_FILE})

add_library(${OM_TRAIN_LIB} STATIC ${SOURCES})

target_include_directories(${OM_TRAIN_LIB} PUBLIC "${OM_INCLUDE_DIR}")
target_link_libraries(${OM_TRAIN_LIB} PUBLIC ${OM_CORE_LIB})

message(STATUS "ONERT MICRO TRAIN BUILD FINISHED")
