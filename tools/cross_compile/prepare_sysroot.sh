#!/usr/bin/env bash
#
# Assemble sysroot for a target board from its SDK.
# Copies headers + libraries for cross-compilation.
#
# Usage:
#   bash tools/cross_compile/prepare_sysroot.sh <board>
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BOARD="${1:-}"
if [ -z "${BOARD}" ]; then
    echo "Usage: prepare_sysroot.sh <board>"
    echo "Available boards:"
    for f in "${SCRIPT_DIR}"/boards/*.conf; do
        basename "${f}" .conf
    done
    exit 1
fi

source "${SCRIPT_DIR}/lib/common.sh"
load_board_conf "${BOARD}" "${SCRIPT_DIR}/boards"

SYSROOT_DIR="${SCRIPT_DIR}/sysroot/${BOARD_SYSROOT_DIR}"

echo "============================================"
echo "  NeuroFabric — ${BOARD_NAME} Sysroot Assembly"
echo "============================================"

# Boards without an SDK (e.g. rpi4) — just create empty sysroot
if [ -z "${BOARD_SDK_SOURCE}" ]; then
    log_warn "No SDK configured for ${BOARD_NAME} — creating empty sysroot"
    mkdir -p "${SYSROOT_DIR}/usr/include"
    mkdir -p "${SYSROOT_DIR}/usr/lib"
    log_info "Location: ${SYSROOT_DIR}"
    exit 0
fi

SDK_DIR="${BOARD_SDK_SOURCE}"
SDK_INCLUDE="${SDK_DIR}/${BOARD_SDK_INCLUDE_SUBDIR}"
SDK_LIB="${SDK_DIR}/${BOARD_SDK_LIB_SUBDIR}"

for dir in "${SDK_INCLUDE}" "${SDK_LIB}"; do
    if [ ! -d "${dir}" ]; then
        log_error "Required directory not found: ${dir}"
        exit 1
    fi
done

# Clean and create
rm -rf "${SYSROOT_DIR}"
mkdir -p "${SYSROOT_DIR}/usr/include"
mkdir -p "${SYSROOT_DIR}/usr/lib"

# Copy headers
log_info "Copying headers..."
for hdr in ${BOARD_SDK_HEADERS}; do
    cp -v "${SDK_INCLUDE}/${hdr}" "${SYSROOT_DIR}/usr/include/" 2>/dev/null || \
        log_warn "Header not found: ${hdr} (skipped)"
done

# Copy libraries
log_info "Copying libraries..."
for lib in ${BOARD_SDK_LIBS}; do
    cp -v "${SDK_LIB}/${lib}" "${SYSROOT_DIR}/usr/lib/" 2>/dev/null || \
        log_warn "Library not found: ${lib} (skipped)"
done

echo ""
echo "============================================"
echo "  Sysroot Assembly Complete"
echo "============================================"
log_info "Location: ${SYSROOT_DIR}"
log_info "Contents:"
ls -la "${SYSROOT_DIR}/usr/include/"
ls -la "${SYSROOT_DIR}/usr/lib/"
TOTAL_SIZE=$(du -sh "${SYSROOT_DIR}" | cut -f1)
log_info "Total size: ${TOTAL_SIZE}"
