# ==================================================================
# Neuro-Fabric — Top-Level CMake Configuration
# ==================================================================
# Minimum 3.21: PROJECT_IS_TOP_LEVEL, improved presets support.
cmake_minimum_required(VERSION 3.21)

project(neurofabric
    VERSION   0.1.1
    LANGUAGES C CXX
)

# ------------------------------------------------------------------
# Global Standards
# ------------------------------------------------------------------
set(CMAKE_C_STANDARD   11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ------------------------------------------------------------------
# Default Build Type (single-config generators only)
# ------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE
        PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

# ------------------------------------------------------------------
# Global Symbol Visibility Policy
#   All shared libraries default to hidden symbols.
#   Only symbols explicitly decorated with NF_API are exported.
# ------------------------------------------------------------------
set(CMAKE_C_VISIBILITY_PRESET     hidden)
set(CMAKE_CXX_VISIBILITY_PRESET   hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# ------------------------------------------------------------------
# Output Directories — keep build artifacts tidy
# ------------------------------------------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ------------------------------------------------------------------
# Options
# ------------------------------------------------------------------
option(NF_BUILD_TESTS    "Build unit tests"          ON)
option(NF_BUILD_EXAMPLES "Build example programs"     ON)
option(NF_PLUGIN_METAL   "Build Apple Metal plugin"   OFF)
option(NF_PLUGIN_RKNN    "Build Rockchip RKNN plugin" OFF)
option(NF_PLUGIN_NETWORK "Build Network Proxy plugin"  ON)
set(NF_TARGET_BOARD "" CACHE STRING "Target board (rk3588, rpi4, ascend)")

# Auto-detect Apple Silicon
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    enable_language(OBJCXX)
    set(NF_PLUGIN_METAL ON CACHE BOOL "" FORCE)
    message(STATUS "[NF] Apple Silicon detected — Metal plugin enabled (OBJCXX)")
endif()

# Board-specific plugin selection
if(NF_TARGET_BOARD STREQUAL "rk3588")
    set(NF_PLUGIN_RKNN ON CACHE BOOL "" FORCE)
    message(STATUS "[NF] Target board: rk3588 — RKNN plugin enabled")
elseif(NF_TARGET_BOARD STREQUAL "ascend")
    # future: set(NF_PLUGIN_ASCEND ON CACHE BOOL "" FORCE)
    message(STATUS "[NF] Target board: ascend")
elseif(NF_TARGET_BOARD STREQUAL "rpi4")
    message(STATUS "[NF] Target board: rpi4 — CPU-only")
elseif(NF_TARGET_BOARD STREQUAL "")
    # Fallback: legacy auto-detect for native on-device builds
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        set(NF_PLUGIN_RKNN ON CACHE BOOL "" FORCE)
        message(STATUS "[NF] Linux aarch64 detected — RKNN plugin enabled (auto)")
    endif()
endif()

# ------------------------------------------------------------------
# Subdirectories
# ------------------------------------------------------------------
add_subdirectory(core)

if(NF_PLUGIN_METAL)
    add_subdirectory(plugins/metal)
endif()

if(NF_PLUGIN_RKNN)
    add_subdirectory(plugins/rknn)
endif()

if(NF_PLUGIN_NETWORK)
    add_subdirectory(plugins/network)
endif()

add_subdirectory(core/src/api)

if(NF_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(NF_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

option(NF_BUILD_TOOLS "Build CLI tools" ON)
if(NF_BUILD_TOOLS)
    add_subdirectory(tools)
endif()
