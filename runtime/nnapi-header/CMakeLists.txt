add_library(nnfw-nnapi-header INTERFACE)
target_include_directories(nnfw-nnapi-header INTERFACE include)


set(NNFW_NNAPI_HEADERS ${CMAKE_CURRENT_LIST_DIR}/include/NeuralNetworks.h
                       ${CMAKE_CURRENT_LIST_DIR}/include/NeuralNetworksEx.h
                       ${CMAKE_CURRENT_LIST_DIR}/include/NeuralNetworksExtensions.h)

install(FILES ${NNFW_NNAPI_HEADERS} DESTINATION include/nnfw)
