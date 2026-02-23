#!/usr/bin/env bash
#
# Build → Deploy to device → Run remotely.
#
# Usage:
#   bash infra/cross_compile/deploy.sh <board> [--skip-build] [--skip-deploy] [-- <remote_args>]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BOARD="${1:-}"
if [ -z "${BOARD}" ]; then
    echo "Usage: deploy.sh <board> [--skip-build] [--skip-deploy] [-- <remote_args>]"
    exit 1
fi
shift

source "${SCRIPT_DIR}/lib/common.sh"
load_board_conf "${BOARD}" "${SCRIPT_DIR}/boards"
load_device_conf "${BOARD}" "${SCRIPT_DIR}/devices"

BUILD_DIR="${REPO_ROOT}/build-${BOARD_NAME}"

SKIP_BUILD=0
SKIP_DEPLOY=0
REMOTE_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build)  SKIP_BUILD=1; shift ;;
        --skip-deploy) SKIP_DEPLOY=1; shift ;;
        --)            shift; REMOTE_ARGS="$*"; break ;;
        *)             REMOTE_ARGS="$*"; break ;;
    esac
done

SSH_TARGET="${TARGET_USER}@${TARGET_HOST}"
SSH_OPTS="-o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -p ${TARGET_PORT}"

echo "============================================"
echo "  NeuroFabric: Build → Deploy → Run (${BOARD_NAME})"
echo "============================================"
echo ""

# ══════════════════════════════════════════════════════════════════
# Step 1: Cross-compile
# ══════════════════════════════════════════════════════════════════
if [ "${SKIP_BUILD}" -eq 0 ]; then
    log_step "Cross-compiling for ${BOARD_NAME}..."
    bash "${SCRIPT_DIR}/build.sh" "${BOARD_NAME}"
else
    log_warn "Skipping build (--skip-build)"
fi

# ══════════════════════════════════════════════════════════════════
# Step 2: Deploy to device
# ══════════════════════════════════════════════════════════════════
if [ "${SKIP_DEPLOY}" -eq 0 ]; then
    log_step "Deploying to ${SSH_TARGET}..."

    # Check connectivity
    if ! ssh ${SSH_OPTS} "${SSH_TARGET}" "echo ok" &>/dev/null; then
        log_error "Cannot reach ${SSH_TARGET}"
        exit 1
    fi

    # Create remote directory
    ssh ${SSH_OPTS} "${SSH_TARGET}" \
        "sudo mkdir -p ${TARGET_DEPLOY_DIR}/lib && sudo chown -R ${TARGET_USER} ${TARGET_DEPLOY_DIR}"

    # Deploy main binary
    if [ -f "${BUILD_DIR}/bin/nf_node_cli" ]; then
        log_info "Deploying nf_node_cli..."
        scp -P "${TARGET_PORT}" "${BUILD_DIR}/bin/nf_node_cli" "${SSH_TARGET}:${TARGET_DEPLOY_DIR}/"
    fi

    # Deploy board-specific plugins
    for plugin in ${BOARD_DEPLOY_PLUGINS}; do
        if [ -f "${BUILD_DIR}/lib/${plugin}" ]; then
            log_info "Deploying ${plugin}..."
            scp -P "${TARGET_PORT}" "${BUILD_DIR}/lib/${plugin}" "${SSH_TARGET}:${TARGET_DEPLOY_DIR}/lib/"
        fi
    done

    log_info "Deploy complete."
else
    log_warn "Skipping deploy (--skip-deploy)"
fi

# ══════════════════════════════════════════════════════════════════
# Step 3: Run on device
# ══════════════════════════════════════════════════════════════════
log_step "Running on ${BOARD_NAME}..."
ssh ${SSH_OPTS} "${SSH_TARGET}" \
    "cd ${TARGET_DEPLOY_DIR} && LD_LIBRARY_PATH=./lib ./nf_node_cli --mode=worker --port=9999 ${REMOTE_ARGS}"
