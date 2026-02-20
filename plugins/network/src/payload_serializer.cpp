/**
 * @file payload_serializer.cpp
 * @brief Zero-Copy Tensor Payload Transport Implementation
 *
 * This is the most performance-critical code in the network plugin.
 * Every byte of tensor data flows through these two functions.
 *
 * Memory flow (send):
 *   [GPU/NPU writes tensor]
 *     → provider.synchronize()     // device execution fence
 *     → cache_sync(FLUSH)          // CPU cache → main memory (ARM)
 *     → map()                      // get CPU-visible VA
 *     → send() in chunks           // kernel copies from VA to NIC
 *     → unmap()
 *
 * Memory flow (recv):
 *   [alloc local nf_buffer]        // DMA-BUF on RK3588, unified on M4
 *     → map()                      // get CPU-visible VA
 *     → recv() in chunks           // NIC → kernel → mapped VA
 *     → unmap()
 *     → cache_sync(FLUSH)          // CPU L1/L2 → main memory
 *     → [NPU/GPU reads tensor]     // device sees coherent data
 */

#include "payload_serializer.h"

#include <cerrno>
#include <cstring>

/* Platform socket includes (already included by network_provider.cpp,
   but this file may be compiled independently) */
#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
  #define WIN32_LEAN_AND_MEAN
  #endif
  #include <winsock2.h>
#else
  #include <sys/socket.h>
  #include <poll.h>
  #include <unistd.h>
#endif

/* ================================================================== */
/*  Internal: Chunked Socket IO with Timeout                           */
/* ================================================================== */

/**
 * Wait for socket readiness with timeout.
 * @param sock      Socket fd.
 * @param for_write true = wait for writable, false = wait for readable.
 * @param timeout_ms Milliseconds to wait. 0 = use default.
 * @return true if ready, false on timeout or error.
 */
static bool wait_ready(int sock, bool for_write, uint32_t timeout_ms) {
    if (timeout_ms == 0) timeout_ms = NF_SOCKET_TIMEOUT_MS;

#ifdef _WIN32
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(static_cast<SOCKET>(sock), &fds);
    struct timeval tv;
    tv.tv_sec  = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    int r = for_write
        ? select(0, nullptr, &fds, nullptr, &tv)
        : select(0, &fds, nullptr, nullptr, &tv);
    return r > 0;
#else
    struct pollfd pfd;
    pfd.fd      = sock;
    pfd.events  = for_write ? POLLOUT : POLLIN;
    pfd.revents = 0;
    int r = ::poll(&pfd, 1, static_cast<int>(timeout_ms));
    return r > 0 && (pfd.revents & pfd.events);
#endif
}

/**
 * Send exactly `len` bytes with chunking and timeout.
 * Handles EAGAIN/EWOULDBLOCK and partial writes.
 */
static nf_status chunked_send(int sock, const uint8_t* data, uint64_t len,
                               uint32_t timeout_ms) {
    uint64_t sent = 0;
    while (sent < len) {
        if (!wait_ready(sock, /*for_write=*/true, timeout_ms)) {
            return NF_ERROR_DEVICE_LOST; /* timeout */
        }

        uint64_t remaining = len - sent;
        /* Cap per-syscall to chunk size to avoid kernel buffer pressure */
        int chunk = static_cast<int>(
            remaining < NF_PAYLOAD_CHUNK_SIZE ? remaining : NF_PAYLOAD_CHUNK_SIZE);

        auto n = ::send(sock, reinterpret_cast<const char*>(data + sent),
                        chunk, 0);
        if (n < 0) {
#ifndef _WIN32
            if (errno == EAGAIN || errno == EWOULDBLOCK) continue;
#endif
            return NF_ERROR_DEVICE_LOST;
        }
        if (n == 0) return NF_ERROR_DEVICE_LOST;
        sent += static_cast<uint64_t>(n);
    }
    return NF_OK;
}

/**
 * Receive exactly `len` bytes with chunking and timeout.
 * Handles EAGAIN/EWOULDBLOCK and partial reads.
 */
static nf_status chunked_recv(int sock, uint8_t* data, uint64_t len,
                               uint32_t timeout_ms) {
    uint64_t got = 0;
    while (got < len) {
        if (!wait_ready(sock, /*for_write=*/false, timeout_ms)) {
            return NF_ERROR_DEVICE_LOST; /* timeout */
        }

        uint64_t remaining = len - got;
        int chunk = static_cast<int>(
            remaining < NF_PAYLOAD_CHUNK_SIZE ? remaining : NF_PAYLOAD_CHUNK_SIZE);

        auto n = ::recv(sock, reinterpret_cast<char*>(data + got), chunk, 0);
        if (n < 0) {
#ifndef _WIN32
            if (errno == EAGAIN || errno == EWOULDBLOCK) continue;
#endif
            return NF_ERROR_DEVICE_LOST;
        }
        if (n == 0) return NF_ERROR_DEVICE_LOST; /* peer closed */
        got += static_cast<uint64_t>(n);
    }
    return NF_OK;
}

/* ================================================================== */
/*  Public API: Send Tensor Payload (Zero-Copy)                        */
/* ================================================================== */

nf_status nf_send_tensor_payload(int sock,
                                 nf_buffer buf,
                                 const nf_buffer_ops* ops,
                                 uint32_t timeout_ms) {
    if (!buf || !ops) return NF_ERROR_INVALID_ARG;

    /* ---- Step 1: Query tensor metadata ---- */
    nf_buffer_info info{};
    if (ops->get_info) {
        nf_status st = ops->get_info(buf, &info);
        if (st != NF_OK) return st;
    }

    uint64_t payload_bytes = info.desc.size_bytes;
    if (payload_bytes == 0) return NF_OK; /* nothing to send */

    /* ---- Step 2: Cache flush (pre-send fence) ---- */
    /*
     * If the producing device (GPU/NPU) just wrote this buffer,
     * we must ensure the data is visible in main memory before
     * the CPU reads it for send().
     *
     * On Apple Silicon (unified memory): hardware-coherent → no-op.
     * On RK3588 (DMA-BUF): MUST flush CPU cache lines that may
     * have been speculatively prefetched or written by DMA.
     */
    if (ops->cache_sync) {
        nf_status st = ops->cache_sync(buf, NF_CACHE_FLUSH, 0, 0);
        if (st != NF_OK) return st;
    }

    /* ---- Step 3: Map buffer → CPU-visible VA ---- */
    void* ptr = nullptr;
    if (!ops->map) return NF_ERROR_UNSUPPORTED_OP;
    {
        nf_status st = ops->map(buf, &ptr);
        if (st != NF_OK) return st;
    }
    if (!ptr) {
        if (ops->unmap) ops->unmap(buf);
        return NF_ERROR_INTERNAL;
    }

    /* ---- Step 4: Chunked send directly from mapped VA ---- */
    /*
     * No intermediate copy. The kernel's send() reads directly
     * from the mapped device/unified memory pointer.
     * On unified memory (M4 Pro), this is the GPU buffer itself.
     * On DMA-BUF (RK3588), this is the mmap'd CMA region.
     */
    nf_status result = chunked_send(sock,
                                     static_cast<const uint8_t*>(ptr),
                                     payload_bytes, timeout_ms);

    /* ---- Step 5: Unmap ---- */
    if (ops->unmap) ops->unmap(buf);

    return result;
}

/* ================================================================== */
/*  Public API: Receive Tensor Payload (Zero-Copy)                     */
/* ================================================================== */

nf_status nf_recv_tensor_payload(int sock,
                                 nf_buffer buf,
                                 const nf_buffer_ops* ops,
                                 const nf_tensor_wire* wire,
                                 uint32_t timeout_ms) {
    if (!buf || !ops || !wire) return NF_ERROR_INVALID_ARG;

    uint64_t payload_bytes = nf_le64toh(wire->payload_bytes);
    if (payload_bytes == 0) return NF_OK;

    /* ---- Step 1: Map buffer → CPU-visible VA ---- */
    /*
     * The buffer was pre-allocated by the local provider
     * (e.g. RKNN's DMA-BUF allocator or Metal's shared buffer).
     * map() gives us a CPU pointer into that device memory.
     */
    void* ptr = nullptr;
    if (!ops->map) return NF_ERROR_UNSUPPORTED_OP;
    {
        nf_status st = ops->map(buf, &ptr);
        if (st != NF_OK) return st;
    }
    if (!ptr) {
        if (ops->unmap) ops->unmap(buf);
        return NF_ERROR_INTERNAL;
    }

    /* ---- Step 2: Chunked recv directly into mapped VA ---- */
    /*
     * recv() writes directly into the device buffer's mapped region.
     * No intermediate copy. On RK3588, this writes into the CMA
     * DMA-BUF mmap region. On M4 Pro, into unified GPU memory.
     */
    nf_status result = chunked_recv(sock,
                                     static_cast<uint8_t*>(ptr),
                                     payload_bytes, timeout_ms);

    /* ---- Step 3: Unmap ---- */
    if (ops->unmap) ops->unmap(buf);

    if (result != NF_OK) return result;

    /* ---- Step 4: Cache flush (post-recv fence) ---- */
    /*
     * CRITICAL on ARM (RK3588):
     *
     * The recv() syscall wrote data through the CPU's store buffer
     * into the mmap'd DMA-BUF region. This data now sits in the
     * CPU's L1/L2 cache. The NPU accesses main memory directly
     * via DMA — it does NOT snoop the CPU cache.
     *
     * We MUST flush (clean) the CPU cache lines to main memory
     * so the NPU reads coherent data. Without this, the NPU
     * reads stale zeros or garbage from DRAM.
     *
     * On Apple Silicon: hardware cache coherency makes this a
     * no-op, but we call it anyway for correctness — the Metal
     * plugin's cache_sync is a no-op function.
     *
     * The correct operation is FLUSH (clean), not INVALIDATE:
     *   FLUSH:      CPU dirty lines → main memory (what we need)
     *   INVALIDATE: discard CPU cache lines (for pre-CPU-read)
     */
    if (ops->cache_sync) {
        result = ops->cache_sync(buf, NF_CACHE_FLUSH, 0, 0);
    }

    return result;
}
