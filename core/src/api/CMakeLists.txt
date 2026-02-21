# ==================================================================
# Neuro-Fabric C API — Shared Library for Python FFI
# ==================================================================

add_library(nf_c_api SHARED
    nf_c_api.cpp
    ${CMAKE_SOURCE_DIR}/core/src/graph/mmap_buffer.cpp
    ${CMAKE_SOURCE_DIR}/core/src/graph/GraphBuilder.cpp
)

target_compile_definitions(nf_c_api PRIVATE NF_C_API_BUILDING)

target_include_directories(nf_c_api
    PUBLIC
        ${CMAKE_SOURCE_DIR}/core/include
    PRIVATE
        ${CMAKE_SOURCE_DIR}/core/src
        ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(nf_c_api PRIVATE nf_core)

set_target_properties(nf_c_api PROPERTIES
    CXX_VISIBILITY_PRESET     hidden
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON
    PREFIX ""
)
