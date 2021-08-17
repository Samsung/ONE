# dummy driver for interface test
set(DUMMY_DRIVER_SRC src/dummy-compile.cpp)
set(HELP_DRIVER_SRC src/help-compile.cpp)
set(DUMMY_PROFILE_SRC src/dummy-profile.cpp)
set(HELP_PROFILE_SRC src/help-profile.cpp)

add_executable(dummy-compile ${DUMMY_DRIVER_SRC})
add_executable(help-compile ${HELP_DRIVER_SRC})
add_executable(dummy-profile ${DUMMY_PROFILE_SRC})
add_executable(help-profile ${HELP_PROFILE_SRC})

set(DUMMY_DRIVER "${CMAKE_CURRENT_BINARY_DIR}/dummy-compile")
set(HELP_DRIVER "${CMAKE_CURRENT_BINARY_DIR}/help-compile")
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
