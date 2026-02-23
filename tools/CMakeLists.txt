# ==================================================================
# Neuro-Fabric — Developer Tools
# ==================================================================

# ---- nf_bench: Benchmark suite (Metal only) ----
if(NF_PLUGIN_METAL)
    add_executable(nf_bench
        nf_bench.cpp
        $<TARGET_OBJECTS:nf_graph_objects>
    )
    target_include_directories(nf_bench PRIVATE
        ${CMAKE_SOURCE_DIR}/core/include
        ${CMAKE_SOURCE_DIR}/core/src
    )
    target_link_libraries(nf_bench PRIVATE nf_core nf_plugin_metal)
endif()
