# ==================================================================
# Neuro-Fabric — CLI Tools
# ==================================================================

if(NF_PLUGIN_RKNN)
    add_executable(nf_node_cli nf_node_cli.cpp)

    target_include_directories(nf_node_cli PRIVATE
        ${CMAKE_SOURCE_DIR}/core/include
    )

    target_link_libraries(nf_node_cli PRIVATE nf_core nf_plugin_rknn)

    if(UNIX AND NOT APPLE)
        target_link_libraries(nf_node_cli PRIVATE dl pthread)
    endif()
    if(APPLE)
        target_link_libraries(nf_node_cli PRIVATE dl)
    endif()

    if(NF_HAS_RKNN_SDK)
        target_compile_definitions(nf_node_cli PRIVATE NF_HAS_RKNN_SDK)
        target_link_libraries(nf_node_cli PRIVATE ${RKNN_RT})
    endif()
endif()
