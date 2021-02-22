###
### This test first generates C++ code from an empty model, and check whether is has compile error
###
set(PREFIX enco-basic-test-000)
set(GENERATED_CPP ${PREFIX}.cpp)
set(GENERATED_ASM ${PREFIX}.embed.S)
set(GENERATED_BIN ${PREFIX}.bin)
set(SOURCE_TARGET ${PREFIX}-src)
set(LIB_TARGET ${PREFIX}-lib)

add_library(${PREFIX}-frontend SHARED enco.test.cpp)
target_link_libraries(${PREFIX}-frontend enco_intf_cmdline)
target_link_libraries(${PREFIX}-frontend enco_intf_frontend)

# NOTE BYPRODUCTS are not specified in order to enforce source code generation
add_custom_command(OUTPUT ${GENERATED_CPP} ${GENERATED_ASM} ${GENERATED_BIN}
                   COMMAND $<TARGET_FILE:enco-cli>
                           --frontend $<TARGET_FILE:${PREFIX}-frontend>
                           --backend-arg ${PREFIX}
                   DEPENDS enco-cli ${PREFIX}-frontend)
set_source_files_properties(${GENERATED_ASM} PROPERTIES GENERATED TRUE LANGUAGE C)
add_library(${LIB_TARGET} SHARED ${GENERATED_CPP} ${GENERATED_ASM})
# NOTE This line is necessary to compile the generated assembly (it includes the generated bin file)
target_include_directories(${LIB_TARGET} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(${LIB_TARGET} PRIVATE ann_api)
