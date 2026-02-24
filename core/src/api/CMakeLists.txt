# ==================================================================
# NeuralOS C API — Shared Library for Python FFI
# ==================================================================

add_library(neuralOS_c_api SHARED
    nf_c_api.cpp
    $<TARGET_OBJECTS:nf_graph_objects>
)

target_compile_definitions(neuralOS_c_api PRIVATE NF_C_API_BUILDING)

target_include_directories(neuralOS_c_api
    PUBLIC
        ${CMAKE_SOURCE_DIR}/core/include
    PRIVATE
        ${CMAKE_SOURCE_DIR}/core/src
        ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(neuralOS_c_api PRIVATE neuralOS_core)

set_target_properties(neuralOS_c_api PROPERTIES
    CXX_VISIBILITY_PRESET     hidden
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON
    PREFIX ""
)
