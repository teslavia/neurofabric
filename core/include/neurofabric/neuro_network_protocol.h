/**
 * @file neuro_network_protocol.h
 * @brief Neuro-Fabric Binary Wire Protocol — Edge-Cloud Tensor Transport
 *
 * Defines the framing format for tensor data and task metadata crossing
 * a network boundary between heterogeneous nodes (e.g. M4 Pro ↔ RK3588).
 *
 * Design constraints:
 *   - All structs are packed (no padding) and little-endian on wire.
 *   - Magic number + CRC32 for frame integrity.
 *   - Metadata (shape, dtype, op_name) travels in the header;
 *     tensor payload follows immediately as raw bytes.
 *   - The receiver deserializes directly into a local nf_buffer
 *     (Metal shared mem on Mac, DMA-BUF on RK3588).
 *
 * Frame layout:
 *   [NF_FRAME_HEADER] [optional NF_TENSOR_WIRE * n_tensors] [payload bytes]
 *
 * Pure C, no C++ dependencies. Safe to include from any plugin.
 */

#ifndef NEUROFABRIC_NETWORK_PROTOCOL_H
#define NEUROFABRIC_NETWORK_PROTOCOL_H

#include "neuro_scheduler_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  1. Constants                                                       */
/* ------------------------------------------------------------------ */

/** Magic bytes: "NF" + version byte + 0x00. Little-endian: 0x0001464E */
#define NF_PROTO_MAGIC       ((uint32_t)0x0001464E)

/** Protocol version — increment on breaking wire format changes. */
#define NF_PROTO_VERSION     ((uint16_t)1)

/** Maximum tensors per frame (inputs + outputs). */
#define NF_PROTO_MAX_TENSORS 24

/* ------------------------------------------------------------------ */
/*  2. Opcodes                                                         */
/* ------------------------------------------------------------------ */

typedef enum nf_proto_opcode {
    /** Client → Server: execute this task remotely. */
    NF_OP_TASK_SUBMIT    = 0x01,

    /** Server → Client: task completed successfully, payload = outputs. */
    NF_OP_TASK_COMPLETE  = 0x02,

    /** Server → Client: task failed, payload = nf_status error code. */
    NF_OP_TASK_ERROR     = 0x03,

    /** Bidirectional: heartbeat / keepalive. No payload. */
    NF_OP_PING           = 0x10,
    NF_OP_PONG           = 0x11,

    /** Client → Server: graceful shutdown request. */
    NF_OP_SHUTDOWN       = 0xFF
} nf_proto_opcode;

/* ------------------------------------------------------------------ */
/*  3. Layout Tags                                                     */
/*     Semantic memory layout for cross-ISA tensor transport.          */
/*     M4 Pro (Metal) defaults to NCHW; RK3588 (RKNN) expects NHWC.   */
/*     The receiver uses this to decide whether reordering is needed.  */
/* ------------------------------------------------------------------ */

typedef enum nf_layout_type {
    NF_LAYOUT_LINEAR = 0,   /**< Row-major contiguous (default, backward compat) */
    NF_LAYOUT_NCHW   = 1,   /**< Batch x Channel x Height x Width               */
    NF_LAYOUT_NHWC   = 2    /**< Batch x Height x Width x Channel               */
} nf_layout_type;

/* ------------------------------------------------------------------ */
/*  4. Tensor Wire Flags                                               */
/*     Per-tensor metadata flags carried in nf_tensor_wire.            */
/* ------------------------------------------------------------------ */

/** Retain tensor in ContextHub across requests (stateful KV cache). */
#define NF_TENSOR_FLAG_STATEFUL  ((uint16_t)0x0001)

/** Payload is int32_t token sequence (prefix key for cache lookup). */
#define NF_TENSOR_FLAG_PREFIX    ((uint16_t)0x0002)

/* ------------------------------------------------------------------ */
/*  5. Tensor Wire Descriptor (packed, no padding)                     */
/*     One per tensor in the frame. Immediately precedes its payload.  */
/* ------------------------------------------------------------------ */

#pragma pack(push, 1)

typedef struct nf_tensor_wire {
    uint8_t   dtype;                    /**< nf_dtype cast to u8       */
    uint8_t   ndim;                     /**< Number of dimensions      */
    uint16_t  layout;                   /**< nf_layout_type cast to u16 */
    uint16_t  flags;                    /**< Bitmask of NF_TENSOR_FLAG_* */
    uint16_t  _pad0;                    /**< Alignment padding         */
    uint64_t  shape[NF_MAX_DIMS];       /**< Shape array               */
    uint64_t  strides[NF_MAX_DIMS];     /**< Byte strides (0=contig)   */
    uint64_t  payload_bytes;            /**< Exact byte count of data  */
} nf_tensor_wire;

/* ------------------------------------------------------------------ */
/*  6. Frame Header (packed)                                           */
/*     Fixed-size prefix for every network message.                    */
/* ------------------------------------------------------------------ */

typedef struct nf_frame_header {
    uint32_t  magic;                    /**< NF_PROTO_MAGIC            */
    uint16_t  version;                  /**< NF_PROTO_VERSION          */
    uint8_t   opcode;                   /**< nf_proto_opcode           */
    uint8_t   flags;                    /**< Reserved for future use   */

    uint64_t  task_id;                  /**< Unique task identifier    */
    uint32_t  seq_num;                  /**< Sequence number for ordering */

    char      op_name[NF_MAX_OP_NAME];  /**< Operator name             */

    uint8_t   n_input_tensors;          /**< Count of input descriptors */
    uint8_t   n_output_tensors;         /**< Count of output descriptors */
    uint16_t  _pad0;

    /**
     * Total payload bytes following the header + tensor descriptors.
     * = sum of all nf_tensor_wire::payload_bytes
     */
    uint64_t  total_payload_bytes;

    /** CRC32 of the header (excluding this field itself). */
    uint32_t  header_crc32;

    uint32_t  _pad1;
} nf_frame_header;

#pragma pack(pop)

/* ------------------------------------------------------------------ */
/*  7. Compile-time size assertions                                    */
/* ------------------------------------------------------------------ */

/** Verify no compiler inserted hidden padding. */
#ifdef __cplusplus
static_assert(sizeof(nf_tensor_wire) ==
    1 + 1 + 2 + 2 + 2 + NF_MAX_DIMS * 8 + NF_MAX_DIMS * 8 + 8,
    "nf_tensor_wire has unexpected padding");

static_assert(sizeof(nf_frame_header) ==
    4 + 2 + 1 + 1 + 8 + 4 + NF_MAX_OP_NAME + 1 + 1 + 2 + 8 + 4 + 4,
    "nf_frame_header has unexpected padding");
#else
_Static_assert(sizeof(nf_tensor_wire) ==
    1 + 1 + 2 + 2 + 2 + NF_MAX_DIMS * 8 + NF_MAX_DIMS * 8 + 8,
    "nf_tensor_wire has unexpected padding");

_Static_assert(sizeof(nf_frame_header) ==
    4 + 2 + 1 + 1 + 8 + 4 + NF_MAX_OP_NAME + 1 + 1 + 2 + 8 + 4 + 4,
    "nf_frame_header has unexpected padding");
#endif

/* ------------------------------------------------------------------ */
/*  8. Endianness Helpers                                              */
/*     Wire format is little-endian. On big-endian hosts these swap.   */
/*     ARM (Apple Silicon, RK3588) and x86 are all LE, so these are   */
/*     typically no-ops — but correctness demands we define them.      */
/* ------------------------------------------------------------------ */

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  static inline uint16_t nf_htole16(uint16_t v) { return __builtin_bswap16(v); }
  static inline uint32_t nf_htole32(uint32_t v) { return __builtin_bswap32(v); }
  static inline uint64_t nf_htole64(uint64_t v) { return __builtin_bswap64(v); }
  static inline uint16_t nf_le16toh(uint16_t v) { return __builtin_bswap16(v); }
  static inline uint32_t nf_le32toh(uint32_t v) { return __builtin_bswap32(v); }
  static inline uint64_t nf_le64toh(uint64_t v) { return __builtin_bswap64(v); }
#else
  static inline uint16_t nf_htole16(uint16_t v) { return v; }
  static inline uint32_t nf_htole32(uint32_t v) { return v; }
  static inline uint64_t nf_htole64(uint64_t v) { return v; }
  static inline uint16_t nf_le16toh(uint16_t v) { return v; }
  static inline uint32_t nf_le32toh(uint32_t v) { return v; }
  static inline uint64_t nf_le64toh(uint64_t v) { return v; }
#endif

/* ------------------------------------------------------------------ */
/*  9. CRC32 (Castagnoli / CRC-32C)                                    */
/*     Minimal software implementation. On Apple Silicon this can be   */
/*     replaced with __builtin_arm_crc32c for hardware acceleration.   */
/* ------------------------------------------------------------------ */

static inline uint32_t nf_crc32c_update(uint32_t crc, const uint8_t* data,
                                        size_t len) {
    crc = ~crc;
    for (size_t i = 0; i < len; ++i) {
        crc ^= data[i];
        for (int j = 0; j < 8; ++j) {
            crc = (crc >> 1) ^ (0x82F63B78u & (-(crc & 1u)));
        }
    }
    return ~crc;
}

static inline uint32_t nf_frame_compute_crc(const nf_frame_header* hdr) {
    /* CRC covers everything except the last 8 bytes (header_crc32 + _pad1). */
    size_t crc_len = sizeof(nf_frame_header) - 8;
    return nf_crc32c_update(0, (const uint8_t*)hdr, crc_len);
}

/* ------------------------------------------------------------------ */
/* 10. Payload Transport Constants & State Machine                     */
/*     Phase 5: chunked tensor payload transfer with timeouts.         */
/* ------------------------------------------------------------------ */

/**
 * Chunk size for large tensor transfers.
 * 256 KB balances syscall overhead vs. memory pressure on edge nodes.
 * Tunable per deployment via NF_CHUNK_SIZE env var at init time.
 */
#define NF_PAYLOAD_CHUNK_SIZE  ((uint64_t)(256 * 1024))

/** Default socket IO timeout in milliseconds. */
#define NF_SOCKET_TIMEOUT_MS   30000

/** Transfer state machine — tracks progress of a multi-chunk payload. */
typedef enum nf_xfer_state {
    NF_XFER_IDLE       = 0,  /**< No transfer in progress             */
    NF_XFER_HEADER     = 1,  /**< Sending/receiving frame header      */
    NF_XFER_TENSOR_DESC= 2,  /**< Sending/receiving tensor descriptors*/
    NF_XFER_PAYLOAD    = 3,  /**< Streaming tensor payload chunks     */
    NF_XFER_COMPLETE   = 4,  /**< Transfer finished successfully      */
    NF_XFER_ERROR      = 5   /**< Transfer failed                     */
} nf_xfer_state;

/**
 * Payload transfer progress — used by both sender and receiver
 * to track multi-chunk streaming state.
 */
typedef struct nf_xfer_progress {
    nf_xfer_state state;
    uint64_t      total_bytes;       /**< Total payload to transfer    */
    uint64_t      transferred_bytes; /**< Bytes completed so far       */
    uint32_t      tensor_index;      /**< Current tensor being xferred */
    uint32_t      n_tensors;         /**< Total tensors in this frame  */
} nf_xfer_progress;

#ifdef __cplusplus
static_assert(sizeof(nf_xfer_progress) == 32,
    "ABI break: nf_xfer_progress size changed");
#else
_Static_assert(sizeof(nf_xfer_progress) == 32,
    "ABI break: nf_xfer_progress size changed");
#endif

#ifdef __cplusplus
}
#endif

#endif /* NEUROFABRIC_NETWORK_PROTOCOL_H */
