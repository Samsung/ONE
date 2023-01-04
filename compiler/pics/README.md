# pics

_pics_ is flatbuffer Python interface for circle schema.

## How to use pics in your module?

Add below lines to your module's `CMakeLists.txt`. It will create a symbolic link to `circle` directory under your module's binary directory.

```
get_target_property(PICS_BIN_PATH pics BINARY_DIR)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/circle
                   COMMAND ${CMAKE_COMMAND} -E create_symlink
                   ${PICS_BIN_PATH}/circle ${CMAKE_CURRENT_BINARY_DIR}/circle)

# Add dependency to ${CMAKE_CURRENT_BINARY_DIR}/circle
```
