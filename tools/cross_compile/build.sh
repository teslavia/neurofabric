#!/usr/bin/env bash
#
# Cross-compile NeuroFabric for a target board via Docker.
#
# Outside Docker: builds image and re-invokes inside container.
# Inside Docker:  runs cmake + make with cross-toolchain.
#
# Usage:
#   bash tools/cross_compile/build.sh <board>
#
# Boards: rk3588, rpi4, ascend
#
# Environment:
#   BUILD_TYPE  — Release (default), Debug, RelWithDebInfo
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BOARD="${1:-}"
if [ -z "${BOARD}" ]; then
    echo "Usage: build.sh <board>"
    echo "Available boards:"
    for f in "${SCRIPT_DIR}"/boards/*.conf; do
        basename "${f}" .conf
    done
    exit 1
fi

source "${SCRIPT_DIR}/lib/common.sh"
load_board_conf "${BOARD}" "${SCRIPT_DIR}/boards"

BUILD_DIR="${REPO_ROOT}/build-${BOARD_NAME}"
TOOLCHAIN_FILE="${SCRIPT_DIR}/toolchains/${BOARD_TOOLCHAIN}"
BUILD_TYPE="${BUILD_TYPE:-Release}"

echo "============================================"
echo "  NeuroFabric ${BOARD_NAME} Cross-Compilation"
echo "============================================"
echo ""
log_info "BOARD      = ${BOARD_NAME}"
log_info "BUILD_TYPE = ${BUILD_TYPE}"

# ══════════════════════════════════════════════════════════════════
# Path A: Outside Docker → build image & re-invoke
# ══════════════════════════════════════════════════════════════════
if [ "${IN_DOCKER:-}" != "1" ]; then

    require_docker

    # Ensure sysroot dir exists (even empty for boards without SDK)
    mkdir -p "${SCRIPT_DIR}/sysroot/${BOARD_SYSROOT_DIR}/usr/include"
    mkdir -p "${SCRIPT_DIR}/sysroot/${BOARD_SYSROOT_DIR}/usr/lib"

    # Build base image if needed
    log_step "Building base Docker image..."
    docker build \
        --build-arg TARGET_ARCH="${BOARD_ARCH}" \
        -t neurofabric-builder-base \
        -f "${SCRIPT_DIR}/docker/Dockerfile.base" \
        "${SCRIPT_DIR}"

    # Build board-specific image
    log_step "Building Docker image '${BOARD_DOCKER_IMAGE}'..."
    docker build \
        -t "${BOARD_DOCKER_IMAGE}" \
        -f "${SCRIPT_DIR}/docker/Dockerfile.${BOARD_NAME}" \
        "${SCRIPT_DIR}"

    log_step "Starting cross-compilation inside Docker..."
    docker run --rm \
        -v "${REPO_ROOT}:/workspace" \
        -e BUILD_TYPE="${BUILD_TYPE}" \
        -e IN_DOCKER=1 \
        "${BOARD_DOCKER_IMAGE}" \
        bash /workspace/tools/cross_compile/build.sh "${BOARD_NAME}"

    echo ""
    echo "============================================"
    echo "  Build Complete"
    echo "============================================"
    if [ -f "${BUILD_DIR}/bin/nf_node_cli" ]; then
        log_info "Binary: ${BUILD_DIR}/bin/nf_node_cli"
        log_info "Size:   $(du -h "${BUILD_DIR}/bin/nf_node_cli" | cut -f1)"
        log_info "Type:   $(file "${BUILD_DIR}/bin/nf_node_cli")"
    else
        log_warn "Binary not found. Check build output above."
    fi
    exit 0
fi

# ══════════════════════════════════════════════════════════════════
# Path B: Inside Docker → cmake + make
# ══════════════════════════════════════════════════════════════════

log_info "Running inside Docker container"
log_info "Toolchain: $(${BOARD_ARCH}-linux-gnu-gcc --version | head -n1)"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

log_step "Running CMake..."
# shellcheck disable=SC2086
cmake /workspace \
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    ${BOARD_CMAKE_FLAGS} \
    -DNF_BUILD_TESTS=OFF \
    -DRKNN_ROOT="${SYSROOT:-}/usr"

NPROC=$(nproc)
log_step "Building (${NPROC} threads)..."
cmake --build . -j"${NPROC}"

log_info "Build complete inside Docker!"
