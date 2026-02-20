# ==================================================================
# Neuro-Fabric — CLI Tools
# ==================================================================

# ---- nf_node_cli: universal coordinator/worker/local executor ----
# Phase 12: no longer requires RKNN — uses GraphBuilder + PipelineEngine
add_executable(nf_node_cli
    nf_node_cli.cpp
    ${CMAKE_SOURCE_DIR}/core/src/graph/mmap_buffer.cpp
    ${CMAKE_SOURCE_DIR}/core/src/graph/GraphBuilder.cpp
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
