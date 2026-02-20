# ==================================================================
# Neuro-Fabric Core — Microkernel Scheduler & Plugin Loader
# ==================================================================
# This library contains ZERO hardware-specific code.
# All device interaction flows through nf_provider_vtable.
# ==================================================================

include(GenerateExportHeader)

# ------------------------------------------------------------------
# Source Collection
# ------------------------------------------------------------------
set(NF_CORE_SOURCES
    src/context.cpp
    src/dag.cpp
    src/plugin_loader.cpp
)

# Platform-specific dynamic library loading implementation.
# Physical isolation: only ONE of these compiles per target OS.
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    list(APPEND NF_CORE_SOURCES src/platform/dl_win32.cpp)
else()
    # POSIX: macOS, Linux (covers Apple Silicon + RK3588)
    list(APPEND NF_CORE_SOURCES src/platform/dl_posix.cpp)
endif()

# ------------------------------------------------------------------
# Library Target
# ------------------------------------------------------------------
add_library(nf_core SHARED ${NF_CORE_SOURCES})

# Generate the cross-platform export macro header automatically.
# This produces: ${CMAKE_CURRENT_BINARY_DIR}/nf_core_export.h
# with macros: NF_CORE_EXPORT, NF_CORE_NO_EXPORT, NF_CORE_DEPRECATED
generate_export_header(nf_core
    BASE_NAME NF_CORE
    EXPORT_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/nf_core_export.h
)

target_compile_definitions(nf_core PRIVATE NF_BUILDING_DLL)

# ------------------------------------------------------------------
# Include Paths
# ------------------------------------------------------------------
target_include_directories(nf_core
    PUBLIC
        # Consumers see the ABI header
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        # Core internals + generated export header
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_BINARY_DIR}
)

# ------------------------------------------------------------------
# Symbol Visibility (belt-and-suspenders with global policy)
# ------------------------------------------------------------------
set_target_properties(nf_core PROPERTIES
    CXX_VISIBILITY_PRESET     hidden
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON
    VERSION                   ${PROJECT_VERSION}
    SOVERSION                 ${PROJECT_VERSION_MAJOR}
)

# ------------------------------------------------------------------
# Platform Link Dependencies
# ------------------------------------------------------------------
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    # Win32 LoadLibrary is in kernel32, linked implicitly
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # dlopen/dlsym live in libdl on Linux
    target_link_libraries(nf_core PRIVATE ${CMAKE_DL_LIBS})
elseif(APPLE)
    # macOS: dlopen is in libSystem, no extra link needed
endif()
