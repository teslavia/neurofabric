# ==================================================================
# Neuro-Fabric — Apple Metal Execution Provider Plugin
# ==================================================================
# Phase 9: Real GPU compute via Objective-C++ (.mm).
# Links Metal + Foundation frameworks. NEVER linked into nf_core.
# ==================================================================

add_library(nf_plugin_metal SHARED
    src/metal_provider.mm
)

target_include_directories(nf_plugin_metal PRIVATE
    ${CMAKE_SOURCE_DIR}/core/include
)

target_compile_definitions(nf_plugin_metal PRIVATE NF_BUILDING_DLL)

set_target_properties(nf_plugin_metal PROPERTIES
    CXX_VISIBILITY_PRESET     hidden
    VISIBILITY_INLINES_HIDDEN ON
    PREFIX ""
    COMPILE_FLAGS "-fobjc-arc"
)

target_link_libraries(nf_plugin_metal PRIVATE
    "-framework Metal"
    "-framework Foundation"
)
