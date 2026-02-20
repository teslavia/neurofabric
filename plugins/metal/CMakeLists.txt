# ==================================================================
# Neuro-Fabric — Apple Metal Execution Provider Plugin
# ==================================================================
# This plugin links against Metal-cpp and MetalPerformanceShaders.
# It is NEVER linked into nf_core — loaded at runtime via dlopen.
# ==================================================================

add_library(nf_plugin_metal SHARED
    src/metal_provider.cpp
)

target_include_directories(nf_plugin_metal PRIVATE
    ${CMAKE_SOURCE_DIR}/core/include
)

target_compile_definitions(nf_plugin_metal PRIVATE NF_BUILDING_DLL)

set_target_properties(nf_plugin_metal PROPERTIES
    CXX_VISIBILITY_PRESET     hidden
    VISIBILITY_INLINES_HIDDEN ON
    PREFIX ""                          # produce nf_plugin_metal.dylib, not libnf_plugin_metal.dylib
)

# Metal frameworks — PRIVATE, never leaks to core
find_library(METAL_FRAMEWORK Metal)
find_library(MPS_FRAMEWORK MetalPerformanceShaders)
find_library(FOUNDATION_FRAMEWORK Foundation)

if(METAL_FRAMEWORK AND MPS_FRAMEWORK)
    target_link_libraries(nf_plugin_metal PRIVATE
        ${METAL_FRAMEWORK}
        ${MPS_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
    )
else()
    message(WARNING "[NF] Metal frameworks not found — plugin will not link")
endif()
