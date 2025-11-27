function(_LLVM_import)

  if(NOT TARGET llvm)
    find_program(llvm_config "llvm-config")
    if (NOT llvm_config)
      return()
    endif(NOT llvm_config)
    message(STATUS "Found llvm-config: ${llvm_config}")

    # get llvm compile options
    execute_process(COMMAND ${llvm_config} --cppflags OUTPUT_VARIABLE
                    LLVM_CPPFLAGS_STR OUTPUT_STRIP_TRAILING_WHITESPACE)
    # split one string to list of option items
    string(REPLACE " " ";" LLVM_CPPFLAGS ${LLVM_CPPFLAGS_STR})
    execute_process(COMMAND ${llvm_config} --has-rtti OUTPUT_VARIABLE
                    LLVM_HAS_RTTI OUTPUT_STRIP_TRAILING_WHITESPACE)
    if("${LLVM_HAS_RTTI}" STREQUAL "NO")
      list(APPEND LLVM_CPPFLAGS "-fno-rtti")
    endif()
    # note: "llvm-config --cxxflags" returns whole string but also includes
    # unwanted "-O3 -DNDEBUG" and several "-Wno-" options so this is not used

    # get llvm link options
    execute_process(COMMAND ${llvm_config} --ldflags OUTPUT_VARIABLE
                    LLVM_LINKFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${llvm_config} --system-libs
                    OUTPUT_VARIABLE LLVM_LINKSYSLIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${llvm_config} --libs core mcjit native
                    OUTPUT_VARIABLE LLVM_LINKLIBS OUTPUT_STRIP_TRAILING_WHITESPACE)

    add_library(llvm INTERFACE)

    foreach(ONE_CPPFLAG ${LLVM_CPPFLAGS})
      target_compile_options(llvm INTERFACE ${ONE_CPPFLAG})
    endforeach()
    target_link_libraries(llvm INTERFACE ${LLVM_LINKFLAGS})
    target_link_libraries(llvm INTERFACE ${LLVM_LINKLIBS})
    target_link_libraries(llvm INTERFACE ${LLVM_LINKSYSLIBS})

  endif(NOT TARGET llvm)

  set(LLVM_FOUND TRUE PARENT_SCOPE)
endfunction(_LLVM_import)

_LLVM_import()
