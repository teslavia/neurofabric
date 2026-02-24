# ==================================================================
# Neuro-Fabric — CUDA Plugin (stub with CPU fallback)
# ==================================================================

add_library(nf_plugin_cuda STATIC
    src/cuda_provider.cpp
)

target_include_directories(nf_plugin_cuda PRIVATE
    ${CMAKE_SOURCE_DIR}/core/include
)

target_link_libraries(nf_plugin_cuda PRIVATE nf_core)

# If real CUDA toolkit is available
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    target_compile_definitions(nf_plugin_cuda PRIVATE NF_HAS_CUDA)
    target_link_libraries(nf_plugin_cuda PRIVATE CUDA::cudart)
endif()
