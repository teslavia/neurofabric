/**
 * @file neuro_ir_format.h
 * @brief Neuro-Fabric IR Binary Format — Zero-Copy Graph Serialization
 *
 * Defines the `.nfir` on-disk format for serialized compute graphs.
 * All structs are packed (no padding), little-endian, and designed for
 * zero-copy deserialization via mmap.
 *
 * File layout:
 *   [nf_ir_header]                        @ 0
 *   [nf_ir_tensor_desc x num_tensors]     @ sizeof(nf_ir_header)
 *   [nf_ir_node_desc   x num_nodes]       @ after tensor descs
 *   [zero padding to 4KB boundary]
 *   [weight payload bytes]                @ payload_offset (4KB aligned)
 *
 * Weight data within the payload is 64-byte aligned for SIMD/DMA.
 */

#ifndef NEUROFABRIC_IR_FORMAT_H
#define NEUROFABRIC_IR_FORMAT_H

#include "neuro_network_protocol.h"   /* nf_crc32c_update, nf_dtype, NF_MAX_DIMS */

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  1. Constants                                                       */
/* ------------------------------------------------------------------ */

/** Magic: "NFIR" in little-endian = 0x5249464E */
#define NF_IR_MAGIC       ((uint32_t)0x5249464E)

/** Format version — increment on breaking layout changes. */
#define NF_IR_VERSION     ((uint32_t)1)

/** Payload region is aligned to this boundary. */
#define NF_IR_PAYLOAD_ALIGN  4096u

/** Individual weights are aligned to this within the payload. */
#define NF_IR_WEIGHT_ALIGN   64u

/** Maximum inputs/outputs per IR node. */
#define NF_IR_MAX_NODE_IO    16u

/* ------------------------------------------------------------------ */
/*  2. Tensor Usage Tags                                               */
/* ------------------------------------------------------------------ */

typedef enum nf_ir_tensor_usage {
    NF_IR_USAGE_ACTIVATION = 0,  /**< Runtime-allocated activation     */
    NF_IR_USAGE_WEIGHT     = 1   /**< mmap'd from weight payload       */
} nf_ir_tensor_usage;

/* ------------------------------------------------------------------ */
/*  3. Packed Structs                                                  */
/* ------------------------------------------------------------------ */

#pragma pack(push, 1)

/**
 * File header — first bytes of every .nfir file.
 */
typedef struct nf_ir_header {
    uint32_t  magic;            /**< NF_IR_MAGIC                       */
    uint32_t  version;          /**< NF_IR_VERSION                     */
    uint32_t  num_tensors;      /**< Count of tensor descriptors       */
    uint32_t  num_nodes;        /**< Count of node descriptors         */
    uint64_t  payload_offset;   /**< Byte offset to weight payload (4KB aligned) */
    uint64_t  payload_size;     /**< Total weight payload bytes        */
    uint32_t  header_crc32;     /**< CRC32C of header (excl this + pad) */
    uint32_t  _pad0;
} nf_ir_header;

/**
 * Tensor descriptor in the IR — describes one named tensor slot.
 */
typedef struct nf_ir_tensor_desc {
    uint32_t  tensor_id;        /**< Unique ID within this graph       */
    uint8_t   dtype;            /**< nf_dtype cast to u8               */
    uint8_t   ndim;             /**< Number of dimensions              */
    uint8_t   usage;            /**< nf_ir_tensor_usage                */
    uint8_t   _pad0;
    uint64_t  shape[NF_MAX_DIMS]; /**< Shape array                    */
    uint64_t  size_bytes;       /**< Total byte size of tensor data    */
    uint64_t  weight_offset;    /**< Offset within payload (WEIGHT only, 64B aligned) */
} nf_ir_tensor_desc;

/**
 * Node descriptor in the IR — one compute operation.
 */
typedef struct nf_ir_node_desc {
    uint32_t  node_id;          /**< Unique ID within this graph       */
    char      op_type[NF_MAX_OP_NAME]; /**< Operator name              */
    uint32_t  n_inputs;         /**< Number of input tensor IDs        */
    uint32_t  n_outputs;        /**< Number of output tensor IDs       */
    uint32_t  input_tensor_ids[NF_IR_MAX_NODE_IO];  /**< Input refs    */
    uint32_t  output_tensor_ids[NF_IR_MAX_NODE_IO]; /**< Output refs   */
    uint32_t  task_flags;       /**< Bitmask of nf_task_flags          */
    uint32_t  _pad0;
} nf_ir_node_desc;

#pragma pack(pop)

/* ------------------------------------------------------------------ */
/*  4. Compile-time size assertions                                    */
/* ------------------------------------------------------------------ */

_Static_assert(sizeof(nf_ir_header) ==
    4 + 4 + 4 + 4 + 8 + 8 + 4 + 4,
    "nf_ir_header has unexpected padding");

_Static_assert(sizeof(nf_ir_tensor_desc) ==
    4 + 1 + 1 + 1 + 1 + NF_MAX_DIMS * 8 + 8 + 8,
    "nf_ir_tensor_desc has unexpected padding");

_Static_assert(sizeof(nf_ir_node_desc) ==
    4 + NF_MAX_OP_NAME + 4 + 4 + NF_IR_MAX_NODE_IO * 4 + NF_IR_MAX_NODE_IO * 4 + 4 + 4,
    "nf_ir_node_desc has unexpected padding");

/* ------------------------------------------------------------------ */
/*  5. CRC Helper                                                      */
/* ------------------------------------------------------------------ */

/** Compute CRC32C of the IR header (excludes header_crc32 + _pad0). */
static inline uint32_t nf_ir_header_compute_crc(const nf_ir_header* hdr) {
    size_t crc_len = sizeof(nf_ir_header) - 8; /* exclude last 8 bytes */
    return nf_crc32c_update(0, (const uint8_t*)hdr, crc_len);
}

#ifdef __cplusplus
}
#endif

#endif /* NEUROFABRIC_IR_FORMAT_H */
