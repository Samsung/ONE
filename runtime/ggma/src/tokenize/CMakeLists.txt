# Tokenize module CMakeLists.txt
# Find SentencePiece if available
nnfw_find_package(SentencePiece QUIET)

set(TOKENIZE_SOURCES
    tokenize.cc
    tokenize.h
    tokenize_factory.h
    tokenize_factory.cc
)

if(SentencePiece_FOUND)
    list(APPEND TOKENIZE_SOURCES tokenize_sentencepiece.cc)
endif()

# Create tokenize as an interface library to export sources and dependencies
add_library(ggma_tokenize INTERFACE)

target_sources(ggma_tokenize INTERFACE ${TOKENIZE_SOURCES})
target_include_directories(ggma_tokenize INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

if(SentencePiece_FOUND)
    target_link_libraries(ggma_tokenize INTERFACE ${SentencePiece_LIBRARIES})
    target_include_directories(ggma_tokenize INTERFACE ${SentencePiece_INCLUDE_DIRS})
    # Add explicit dependency to ensure SentencePiece is built and installed when ggma is built
    add_dependencies(ggma_tokenize sentencepiece)
endif()
