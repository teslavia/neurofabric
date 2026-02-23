# ==================================================================
# Neuro-Fabric — Application Layer (Layer 1)
# ==================================================================

# ---- nf_node_cli: universal coordinator/worker/local executor ----
add_executable(nf_node_cli
    nf_node_cli.cpp
    $<TARGET_OBJECTS:nf_graph_objects>
)

target_include_directories(nf_node_cli PRIVATE
    ${CMAKE_SOURCE_DIR}/core/include
    ${CMAKE_SOURCE_DIR}/core/src          # graph/mmap_buffer.h (internal)
)

target_link_libraries(nf_node_cli PRIVATE nf_core)

if(UNIX AND NOT APPLE)
    target_link_libraries(nf_node_cli PRIVATE dl pthread)
endif()
if(APPLE)
    target_link_libraries(nf_node_cli PRIVATE dl)
endif()

# Optional: link RKNN plugin when available (for real NPU dispatch)
if(NF_PLUGIN_RKNN)
    target_link_libraries(nf_node_cli PRIVATE nf_plugin_rknn)
    target_compile_definitions(nf_node_cli PRIVATE NF_HAS_RKNN_SDK)
    if(NF_HAS_RKNN_SDK)
        target_link_libraries(nf_node_cli PRIVATE ${RKNN_RT})
    endif()
endif()

# ---- Metal-dependent apps ----
if(NF_PLUGIN_METAL)
    # ---- nf_generate: text generation CLI ----
    add_executable(nf_generate
        nf_generate.cpp
        $<TARGET_OBJECTS:nf_graph_objects>
    )
    target_include_directories(nf_generate PRIVATE
        ${CMAKE_SOURCE_DIR}/core/include
        ${CMAKE_SOURCE_DIR}/core/src
    )
    target_link_libraries(nf_generate PRIVATE nf_core nf_plugin_metal nf_model_headers)

    # ---- nf_serve: OpenAI-compatible HTTP server ----
    add_executable(nf_serve
        nf_serve.cpp
        $<TARGET_OBJECTS:nf_graph_objects>
    )
    target_include_directories(nf_serve PRIVATE
        ${CMAKE_SOURCE_DIR}/core/include
        ${CMAKE_SOURCE_DIR}/core/src
    )
    target_link_libraries(nf_serve PRIVATE nf_core nf_plugin_metal nf_model_headers)
endif()
