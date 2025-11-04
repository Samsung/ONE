# Tokenize module CMakeLists.txt
# Find SentencePiece if available
nnfw_find_package(SentencePiece QUIET)

set(TOKENIZE_SOURCES
    Tokenizer.h
    TokenizerFactory.h
    TokenizerFactory.cc
)

if(SentencePiece_FOUND)
    list(APPEND TOKENIZE_SOURCES TokenizerSentencePiece.cc)
endif()

# Create tokenize as a shared library
add_library(ggma_tokenize SHARED ${TOKENIZE_SOURCES})

target_include_directories(ggma_tokenize PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../include)

if(SentencePiece_FOUND)
    target_link_libraries(ggma_tokenize PRIVATE ${SentencePiece_LIBRARIES})
    target_include_directories(ggma_tokenize PRIVATE ${SentencePiece_INCLUDE_DIRS})
    # Add explicit dependency to ensure SentencePiece is built and installed when ggma is built
    add_dependencies(ggma_tokenize sentencepiece)

    # Add log library for Android
    if(ANDROID)
        target_link_libraries(ggma_tokenize PUBLIC log)
    endif()
endif()
