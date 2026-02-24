# ==================================================================
# NeuralOS — Rockchip RKNN Execution Provider Plugin
# ==================================================================
# Links against librknnrt when available. Falls back to simulation.
# Loaded at runtime, never linked into core.
# ==================================================================

add_library(nf_plugin_rknn SHARED
    src/rknn_provider.cpp
)

target_include_directories(nf_plugin_rknn PRIVATE
    ${CMAKE_SOURCE_DIR}/core/include
)

target_compile_definitions(nf_plugin_rknn PRIVATE NF_BUILDING_DLL)

set_target_properties(nf_plugin_rknn PROPERTIES
    CXX_VISIBILITY_PRESET     hidden
    VISIBILITY_INLINES_HIDDEN ON
    PREFIX ""
)

# RKNN runtime — user must set RKNN_ROOT or install to system paths
find_library(RKNN_RT rknnrt
    HINTS ENV RKNN_ROOT
          /usr/lib/aarch64-linux-gnu
          ${CMAKE_SYSROOT}/usr/lib
)
find_path(RKNN_INCLUDE_DIR rknn_api.h
    HINTS ENV RKNN_ROOT
          /usr/include
          ${CMAKE_SYSROOT}/usr/include
    PATH_SUFFIXES include
)
if(RKNN_RT)
    target_link_libraries(nf_plugin_rknn PRIVATE ${RKNN_RT})
    target_compile_definitions(nf_plugin_rknn PRIVATE NF_HAS_RKNN_SDK)
    set(NF_HAS_RKNN_SDK ON CACHE BOOL "RKNN SDK found" FORCE)
    if(RKNN_INCLUDE_DIR)
        target_include_directories(nf_plugin_rknn PRIVATE ${RKNN_INCLUDE_DIR})
    endif()
    message(STATUS "[NF] RKNN SDK found — real NPU path enabled")
    message(STATUS "[NF] RKNN include: ${RKNN_INCLUDE_DIR}")
else()
    set(NF_HAS_RKNN_SDK OFF CACHE BOOL "RKNN SDK found" FORCE)
    message(WARNING "[NF] librknnrt not found — RKNN plugin uses simulation mode")
endif()
