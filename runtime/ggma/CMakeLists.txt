if (NOT BUILD_GGMA_API)
  return()
endif()

file(GLOB_RECURSE API_SRC "src/*.cc")
list(FILTER API_SRC EXCLUDE REGEX "src/tokenize/.*")

set(GGMA_DEV ggma-dev)
add_library(${GGMA_DEV} SHARED ${API_SRC})

# Public headers to publish
set(GGMA_API_HEADERS include/ggma_api.h include/ggma_macro.h include/ggma_types.h include/ggma_tokenize.h
                     include/ggma_context.h include/ggma_generate.h)

# GGMA install directory (same as ONERT_INSTALL_APIDIR)
set(GGMA_INSTALL_LIBDIR ${CMAKE_INSTALL_LIBDIR}/ggma)
set(GGMA_INSTALL_APIDIR ${CMAKE_INSTALL_LIBDIR}/ggma)

target_link_libraries(${GGMA_DEV} PRIVATE jsoncpp ${LIB_PTHREAD})
target_link_libraries(${GGMA_DEV} PRIVATE nnfw-dev)

# Add tokenize subdirectory
add_subdirectory(src/tokenize)

# Link tokenize library
target_link_libraries(${GGMA_DEV} PUBLIC ggma_tokenize)
# NOTE Below line is added to remove warning for android build
#      It will be removed after android build uses gold linker
if (ANDROID)
  target_link_libraries(${GGMA_DEV} INTERFACE log)
endif (ANDROID)

target_include_directories(${GGMA_DEV} PUBLIC include)

# GGMA RPATH configuration: find ggma libraries
set(GGMA_RPATH_API "$ORIGIN:$ORIGIN/..")

set_target_properties(${GGMA_DEV} PROPERTIES
  PUBLIC_HEADER "${GGMA_API_HEADERS}"
  OUTPUT_NAME ggma_api
  INSTALL_RPATH ${GGMA_RPATH_API})

install(TARGETS ${GGMA_DEV}
        LIBRARY DESTINATION ${GGMA_INSTALL_APIDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ggma)

# Install pkg-config file for GGMA API
configure_file(ggma.pc.in ggma.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ggma.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
