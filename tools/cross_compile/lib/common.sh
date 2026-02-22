#!/usr/bin/env bash
#
# Shared shell library for NeuroFabric build/deploy scripts.
# Source this file — do not execute directly.
#

# ── Color codes & logging ──────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()  { echo -e "${CYAN}[STEP]${NC}  $1"; }

# ── load_board_conf <board> <boards_dir> ───────────────────────────
# Validates and sources boards/<board>.conf, sets BOARD_* vars.
load_board_conf() {
    local board="$1"
    local boards_dir="$2"
    local conf="${boards_dir}/${board}.conf"
    if [ ! -f "${conf}" ]; then
        log_error "Board config not found: ${conf}"
        exit 1
    fi
    source "${conf}"
}

# ── load_device_conf <board> <devices_dir> ─────────────────────────
# Validates and sources devices/<board>.conf, sets TARGET_* vars.
load_device_conf() {
    local board="$1"
    local devices_dir="$2"
    local conf="${devices_dir}/${board}.conf"
    if [ ! -f "${conf}" ]; then
        log_error "Device config not found: ${conf}"
        exit 1
    fi
    source "${conf}"
}

# ── require_docker ─────────────────────────────────────────────────
# Checks that docker CLI exists and daemon is running.
require_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Install Docker Desktop."
        exit 1
    fi
    if ! docker info &> /dev/null 2>&1; then
        log_error "Docker daemon not running. Start Docker Desktop."
        exit 1
    fi
    log_info "Docker: $(docker --version)"
}
