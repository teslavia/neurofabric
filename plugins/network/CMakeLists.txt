# ==================================================================
# NeuralOS — Network Proxy Execution Provider Plugin
# ==================================================================
# Implements nf_provider_vtable over TCP sockets.
# Loaded at runtime via dlopen, never linked into neuralOS_core.
# ==================================================================

add_library(nf_plugin_network SHARED
    src/network_provider.cpp
    src/payload_serializer.cpp
)

target_include_directories(nf_plugin_network PRIVATE
    ${CMAKE_SOURCE_DIR}/core/include
)

target_compile_definitions(nf_plugin_network PRIVATE NF_BUILDING_DLL)

set_target_properties(nf_plugin_network PROPERTIES
    CXX_VISIBILITY_PRESET     hidden
    VISIBILITY_INLINES_HIDDEN ON
    PREFIX ""
)

# Platform socket libraries
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_link_libraries(nf_plugin_network PRIVATE ws2_32)
endif()
# POSIX (macOS, Linux): sockets are in libc, no extra link needed.
