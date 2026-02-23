/**
 * @file payload_serializer.h
 * @brief Zero-Copy Tensor Payload Transport — Send & Receive Engine
 *
 * The critical data path for edge-cloud tensor transfer.
 * All functions operate directly on mapped nf_buffer memory —
 * no intermediate std::vector copies.
 *
 * Send path:  sync_cache(FLUSH) → map → send chunks → unmap
 * Recv path:  alloc → map → recv chunks → unmap → sync_cache(FLUSH)
 *
 * Cache coherency contract:
 *   SENDER:  After GPU/NPU writes, flush cache before network read.
 *   RECEIVER: After network write into DMA-BUF, flush cache before
 *             handing buffer to NPU (ARM L1/L2 → main memory).
 */

#ifndef NF_PAYLOAD_SERIALIZER_H
#define NF_PAYLOAD_SERIALIZER_H

#include "neurofabric/abi/neuro_network_protocol.h"
#include "neurofabric/abi/neuro_buffer_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Send a single tensor's payload over the socket.
 * Zero-copy: maps the buffer, sends directly from mapped VA.
 *
 * Caller must ensure the producing device (GPU/NPU) has finished
 * writing before calling this. The function handles cache flush.
 *
 * @param sock       Connected socket fd.
 * @param buf        Buffer handle to send.
 * @param ops        Buffer operations vtable.
 * @param timeout_ms Per-chunk timeout (0 = NF_SOCKET_TIMEOUT_MS default).
 * @return NF_OK on success.
 */
nf_status nf_send_tensor_payload(int sock,
                                 nf_buffer buf,
                                 const nf_buffer_ops* ops,
                                 uint32_t timeout_ms);

/**
 * Receive a tensor's payload from the socket into a pre-allocated buffer.
 * Zero-copy: maps the buffer, receives directly into mapped VA.
 *
 * After recv completes, flushes CPU cache so the device (NPU/GPU)
 * sees coherent data.
 *
 * @param sock       Connected socket fd.
 * @param buf        Pre-allocated buffer to receive into.
 * @param ops        Buffer operations vtable.
 * @param wire       Tensor wire descriptor (carries expected byte count).
 * @param timeout_ms Per-chunk timeout (0 = NF_SOCKET_TIMEOUT_MS default).
 * @return NF_OK on success.
 */
nf_status nf_recv_tensor_payload(int sock,
                                 nf_buffer buf,
                                 const nf_buffer_ops* ops,
                                 const nf_tensor_wire* wire,
                                 uint32_t timeout_ms);

#ifdef __cplusplus
}
#endif

#endif /* NF_PAYLOAD_SERIALIZER_H */
