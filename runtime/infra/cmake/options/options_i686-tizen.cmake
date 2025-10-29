#
# i686 tizen cmake options
#
option(BUILD_TENSORFLOW_LITE "Build TensorFlow Lite from the downloaded source" OFF)
option(DOWNLOAD_GTEST "Download Google Test source and build Google Test" OFF)

option(ENVVAR_ONERT_CONFIG "Use environment variable for onert configuration" OFF)

option(BUILD_XNNPACK "Build XNNPACK" OFF)

option(BUILD_GGMA_API "Build GGMA API for Generative AI" OFF)
option(BUILD_GGMA_RUN "Build ggma_run test driver" OFF)
option(DOWNLOAD_SENTENCEPIECE "Download SentencePiece source" OFF)
option(BUILD_SENTENCEPIECE "Build SentencePiece library from the source" OFF)

