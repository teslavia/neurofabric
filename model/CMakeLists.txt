# ==================================================================
# Neuro-Fabric — Model Layer (Layer 2): header-only model libraries
# ==================================================================

add_library(nf_model_headers INTERFACE)
target_include_directories(nf_model_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
