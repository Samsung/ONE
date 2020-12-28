# dummy driver for interface test
set(DUMMY_DRIVER_SRC src/dummy-compile.cpp)

add_executable(dummy-compile ${DUMMY_DRIVER_SRC})

set(DUMMY_DRIVER "${CMAKE_CURRENT_BINARY_DIR}/dummy-compile")

install(FILES ${DUMMY_DRIVER}
        PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE
        DESTINATION test)
