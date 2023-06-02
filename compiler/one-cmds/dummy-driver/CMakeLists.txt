# dummy driver for interface test
set(DUMMY_DRIVER_SRC src/dummy-compile.cpp)
set(DUMMY_V2_DRIVER_SRC src/dummyV2-compile.cpp)
set(HELP_DRIVER_SRC src/help-compile.cpp)
set(DUMMY_INFER_SRC src/dummy-infer.cpp)
set(DUMMY_INFER_V2_SRC src/dummy-inferV2.cpp)
set(HELP_INFER_SRC src/help-infer.cpp)
set(DUMMY_PROFILE_SRC src/dummy-profile.cpp)
set(DUMMY_V2_PROFILE_SRC src/dummyV2-profile.cpp)
set(HELP_PROFILE_SRC src/help-profile.cpp)
set(DUMMY_ENV_SRC src/dummyEnv-compile.cpp)
set(DUMMY_ONNX_EXT src/dummy-onnx-ext.cpp)

add_executable(dummy-compile ${DUMMY_DRIVER_SRC})
add_executable(dummyV2-compile ${DUMMY_V2_DRIVER_SRC})
add_executable(help-compile ${HELP_DRIVER_SRC})
add_executable(dummy-infer ${DUMMY_INFER_SRC})
add_executable(dummy-inferV2 ${DUMMY_INFER_V2_SRC})
add_executable(help-infer ${HELP_INFER_SRC})
add_executable(dummy-profile ${DUMMY_PROFILE_SRC})
add_executable(dummyV2-profile ${DUMMY_V2_PROFILE_SRC})
add_executable(help-profile ${HELP_PROFILE_SRC})
add_executable(dummyEnv-compile ${DUMMY_ENV_SRC})
add_executable(dummy-onnx-ext ${DUMMY_ONNX_EXT})

set(DUMMY_DRIVER "${CMAKE_CURRENT_BINARY_DIR}/dummy-compile")
set(DUMMY_V2_DRIVER "${CMAKE_CURRENT_BINARY_DIR}/dummyV2-compile")
set(HELP_DRIVER "${CMAKE_CURRENT_BINARY_DIR}/help-compile")
set(DUMMY_INFER "${CMAKE_CURRENT_BINARY_DIR}/dummy-infer")
set(DUMMY_INFER_V2 "${CMAKE_CURRENT_BINARY_DIR}/dummy-inferV2")
set(HELP_INFER "${CMAKE_CURRENT_BINARY_DIR}/help-infer")
set(DUMMY_PROFILE "${CMAKE_CURRENT_BINARY_DIR}/dummy-profile")
set(DUMMY_V2_PROFILE "${CMAKE_CURRENT_BINARY_DIR}/dummyV2-profile")
set(HELP_PROFILE "${CMAKE_CURRENT_BINARY_DIR}/help-profile")
set(DUMMY_ENV "${CMAKE_CURRENT_BINARY_DIR}/dummyEnv-compile")
set(DUMMY_ONNX_EXT "${CMAKE_CURRENT_BINARY_DIR}/dummy-onnx-ext")

install(FILES ${DUMMY_DRIVER}
        PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE
        DESTINATION test)

install(FILES ${DUMMY_V2_DRIVER}
        PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE
        DESTINATION test)

install(FILES ${HELP_DRIVER}
        PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE
        DESTINATION test)

install(FILES ${DUMMY_INFER}
        PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE
        DESTINATION test)

install(FILES ${DUMMY_INFER_V2}
        PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE
        DESTINATION test)

install(FILES ${HELP_INFER}
        PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE
        DESTINATION test)

install(FILES ${DUMMY_PROFILE}
        PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE
        DESTINATION test)

install(FILES ${DUMMY_V2_PROFILE}
        PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE
        DESTINATION test)

install(FILES ${HELP_PROFILE}
        PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE
        DESTINATION test)

install(FILES ${DUMMY_ENV}
        PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE
        DESTINATION test)

install(FILES ${DUMMY_ONNX_EXT}
        PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE
        DESTINATION test)
