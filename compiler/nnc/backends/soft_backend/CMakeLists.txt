set(SOFT_BACKEND_CPP_SOURCES CPPGenerator.cpp ModelAnalyzer.cpp SBSerializer.cpp SequencedIR.cpp)

file(GLOB_RECURSE SOFT_DEF_SOURCES "*.def")

nnc_make_generated_sources("${SOFT_DEF_SOURCES}" ${CMAKE_CURRENT_BINARY_DIR} SOFT_GENERATED_SOURCES)

nnc_add_library(soft_backend_cpp SHARED ${SOFT_BACKEND_CPP_SOURCES} ${SOFT_GENERATED_SOURCES})
target_include_directories(soft_backend_cpp PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(soft_backend_cpp PRIVATE mir)
if(ONE_UBUNTU_CODENAME_BIONIC)
    target_link_libraries(soft_backend_cpp PRIVATE stdc++fs)
endif()

# install soft backend c++ library
nnc_install_library(soft_backend_cpp)
