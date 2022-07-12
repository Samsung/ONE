# dummy driver for interface test
set(DUMMY_DRIVER_SRC src/dummy-compile.cpp)
set(HELP_DRIVER_SRC src/help-compile.cpp)
set(DUMMY_INFER_SRC src/dummy-infer.cpp)
set(DUMMY_INFER_V2_SRC src/dummy-inferV2.cpp)
set(HELP_INFER_SRC src/help-infer.cpp)
set(DUMMY_INIT_SRC src/dummy-init.cpp)
set(HELP_INIT_SRC src/help-init.cpp)
set(DUMMY_PROFILE_SRC src/dummy-profile.cpp)
set(HELP_PROFILE_SRC src/help-profile.cpp)

add_executable(dummy-compile ${DUMMY_DRIVER_SRC})
add_executable(help-compile ${HELP_DRIVER_SRC})
add_executable(dummy-infer ${DUMMY_INFER_SRC})
add_executable(dummy-inferV2 ${DUMMY_INFER_V2_SRC})
add_executable(help-infer ${HELP_INFER_SRC})
add_executable(dummy-init ${DUMMY_INIT_SRC})
add_executable(help-init ${HELP_INIT_SRC})
add_executable(dummy-profile ${DUMMY_PROFILE_SRC})
add_executable(help-profile ${HELP_PROFILE_SRC})

set(DUMMY_DRIVER "${CMAKE_CURRENT_BINARY_DIR}/dummy-compile")
set(HELP_DRIVER "${CMAKE_CURRENT_BINARY_DIR}/help-compile")
set(DUMMY_INFER "${CMAKE_CURRENT_BINARY_DIR}/dummy-infer")
set(DUMMY_INFER_V2 "${CMAKE_CURRENT_BINARY_DIR}/dummy-inferV2")
set(HELP_INFER "${CMAKE_CURRENT_BINARY_DIR}/help-infer")
set(DUMMY_INIT "${CMAKE_CURRENT_BINARY_DIR}/dummy-init")
set(HELP_INIT "${CMAKE_CURRENT_BINARY_DIR}/help-init")
set(DUMMY_PROFILE "${CMAKE_CURRENT_BINARY_DIR}/dummy-profile")
set(HELP_PROFILE "${CMAKE_CURRENT_BINARY_DIR}/help-profile")

install(FILES ${DUMMY_DRIVER}
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

install(FILES ${DUMMY_INIT}
        PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE
        DESTINATION test)

install(FILES ${HELP_INIT}
        PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE
        DESTINATION test)

install(FILES ${DUMMY_PROFILE}
        PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE
        DESTINATION test)

install(FILES ${HELP_PROFILE}
        PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE
        DESTINATION test)
