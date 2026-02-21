"""
Neuro-Fabric AOT Compiler — Python .nfir Binary IR Emitter

Generates .nfir files byte-compatible with neuro_ir_format.h.
Zero dependencies beyond stdlib + numpy.

Binary layout:
  [nf_ir_header]                      @ 0          (40 bytes)
  [nf_ir_tensor_desc x num_tensors]   @ 40         (88 bytes each)
  [nf_ir_node_desc   x num_nodes]     @ after tensors (212 bytes each)
  [zero padding to 4KB boundary]
  [weight payload]                    @ payload_offset (4KB aligned)
"""

import struct
import sys
from typing import List, Optional, Tuple

import numpy as np

# ── Constants (mirror neuro_ir_format.h) ──────────────────────────

NF_IR_MAGIC         = 0x5249464E
NF_IR_VERSION       = 1
NF_IR_PAYLOAD_ALIGN = 4096
NF_IR_WEIGHT_ALIGN  = 64
NF_MAX_DIMS         = 8
NF_MAX_OP_NAME      = 64
NF_IR_MAX_NODE_IO   = 16

HEADER_SIZE      = 40   # 4+4+4+4+8+8+4+4
TENSOR_DESC_SIZE = 88   # 4+1+1+1+1+64+8+8
NODE_DESC_SIZE   = 212  # 4+64+4+4+64+64+4+4

# nf_ir_tensor_usage
USAGE_ACTIVATION = 0
USAGE_WEIGHT     = 1

# nf_dtype
DTYPE_F32  = 0
DTYPE_F16  = 1
DTYPE_BF16 = 2
DTYPE_I8   = 3
DTYPE_I32  = 4
DTYPE_U8   = 5

_DTYPE_BYTES = {DTYPE_F32: 4, DTYPE_F16: 2, DTYPE_BF16: 2,
                DTYPE_I8: 1, DTYPE_I32: 4, DTYPE_U8: 1}

# nf_task_flags
TASK_NONE   = 0
TASK_ASYNC  = 0x01
TASK_FENCE  = 0x02
TASK_REMOTE = 0x04

# ── CRC32C (Castagnoli) — matches nf_crc32c_update exactly ────────

CRC32C_POLY = 0x82F63B78

def crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            mask = -(crc & 1) & 0xFFFFFFFF
            crc = ((crc >> 1) ^ (CRC32C_POLY & mask)) & 0xFFFFFFFF
    return crc ^ 0xFFFFFFFF


def _align_up(val: int, align: int) -> int:
    return (val + align - 1) & ~(align - 1)


# ── TensorDesc ────────────────────────────────────────────────────

class TensorDesc:
    """Mirrors nf_ir_tensor_desc (88 bytes packed)."""

    def __init__(self, tensor_id: int, dtype: int, shape: Tuple[int, ...],
                 usage: int = USAGE_ACTIVATION,
                 weight_data: Optional[np.ndarray] = None):
        self.tensor_id = tensor_id
        self.dtype = dtype
        self.ndim = len(shape)
        self.usage = usage
        self.shape = shape
        self.size_bytes = int(np.prod(shape)) * _DTYPE_BYTES[dtype]
        self.weight_offset = 0  # filled during layout
        self.weight_data = weight_data

    def pack(self) -> bytes:
        shape_padded = list(self.shape) + [0] * (NF_MAX_DIMS - self.ndim)
        return struct.pack(
            '<IBBBB' + 'Q' * NF_MAX_DIMS + 'QQ',
            self.tensor_id,
            self.dtype, self.ndim, self.usage, 0,
            *shape_padded,
            self.size_bytes,
            self.weight_offset,
        )


# ── NodeDesc ──────────────────────────────────────────────────────

class NodeDesc:
    """Mirrors nf_ir_node_desc (212 bytes packed)."""

    def __init__(self, node_id: int, op_type: str,
                 input_ids: List[int], output_ids: List[int],
                 task_flags: int = TASK_NONE):
        self.node_id = node_id
        self.op_type = op_type
        self.n_inputs = len(input_ids)
        self.n_outputs = len(output_ids)
        self.input_ids = input_ids
        self.output_ids = output_ids
        self.task_flags = task_flags

    def pack(self) -> bytes:
        op_bytes = self.op_type.encode('ascii')[:NF_MAX_OP_NAME]
        op_bytes = op_bytes.ljust(NF_MAX_OP_NAME, b'\x00')
        in_padded = self.input_ids + [0] * (NF_IR_MAX_NODE_IO - self.n_inputs)
        out_padded = self.output_ids + [0] * (NF_IR_MAX_NODE_IO - self.n_outputs)
        return struct.pack(
            '<I64sII' + 'I' * NF_IR_MAX_NODE_IO + 'I' * NF_IR_MAX_NODE_IO + 'II',
            self.node_id,
            op_bytes,
            self.n_inputs, self.n_outputs,
            *in_padded,
            *out_padded,
            self.task_flags, 0,
        )


# ── NfirBuilder — the core compiler ───────────────────────────────

class NfirBuilder:
    """Programmatic API for building .nfir files."""

    def __init__(self):
        self.tensors: List[TensorDesc] = []
        self.nodes: List[NodeDesc] = []

    def add_tensor(self, tensor_id: int, dtype: int, shape: Tuple[int, ...],
                   usage: int = USAGE_ACTIVATION,
                   weight_data: Optional[np.ndarray] = None) -> 'NfirBuilder':
        self.tensors.append(TensorDesc(tensor_id, dtype, shape, usage, weight_data))
        return self

    def add_node(self, node_id: int, op_type: str,
                 input_ids: List[int], output_ids: List[int],
                 task_flags: int = TASK_NONE) -> 'NfirBuilder':
        self.nodes.append(NodeDesc(node_id, op_type, input_ids, output_ids, task_flags))
        return self

    def build(self, path: str) -> None:
        """Emit .nfir binary file."""
        num_tensors = len(self.tensors)
        num_nodes = len(self.nodes)

        # ── Step 1: Compute metadata size and payload offset ──
        metadata_size = HEADER_SIZE + num_tensors * TENSOR_DESC_SIZE + num_nodes * NODE_DESC_SIZE
        payload_offset = _align_up(metadata_size, NF_IR_PAYLOAD_ALIGN)

        # ── Step 2: Layout weight offsets within payload ──
        weight_blobs: List[bytes] = []
        current_offset = 0
        for td in self.tensors:
            if td.usage == USAGE_WEIGHT and td.weight_data is not None:
                aligned = _align_up(current_offset, NF_IR_WEIGHT_ALIGN)
                padding = aligned - current_offset
                if padding > 0:
                    weight_blobs.append(b'\x00' * padding)
                    current_offset = aligned
                td.weight_offset = current_offset
                raw = td.weight_data.tobytes()
                weight_blobs.append(raw)
                current_offset += len(raw)

        payload_data = b''.join(weight_blobs)
        payload_size = len(payload_data)

        # ── Step 3: Pack header (CRC computed over first 32 bytes) ──
        header_no_crc = struct.pack(
            '<IIIIQQ',
            NF_IR_MAGIC, NF_IR_VERSION,
            num_tensors, num_nodes,
            payload_offset, payload_size,
        )
        assert len(header_no_crc) == 32
        header_crc = crc32c(header_no_crc)
        header = header_no_crc + struct.pack('<II', header_crc, 0)
        assert len(header) == HEADER_SIZE

        # ── Step 4: Pack tensor + node descriptors ──
        tensor_data = b''.join(td.pack() for td in self.tensors)
        node_data = b''.join(nd.pack() for nd in self.nodes)

        # ── Step 5: Zero-padding to payload_offset ──
        written = HEADER_SIZE + len(tensor_data) + len(node_data)
        pad_size = payload_offset - written
        assert pad_size >= 0

        # ── Step 6: Write file ──
        with open(path, 'wb') as f:
            f.write(header)
            f.write(tensor_data)
            f.write(node_data)
            f.write(b'\x00' * pad_size)
            f.write(payload_data)


# ── CLI entry point ───────────────────────────────────────────────

def _demo_graph(output: str, remote_nodes: Optional[str] = None):
    """Generate a demo 2-node graph: weighted_add → mock_relu."""
    n_floats = 1024
    weights = np.arange(n_floats, dtype=np.float32) * 0.1

    remote_set = set()
    if remote_nodes:
        remote_set = {int(x) for x in remote_nodes.split(',')}

    builder = NfirBuilder()
    builder.add_tensor(0, DTYPE_F32, (n_floats,), USAGE_WEIGHT, weight_data=weights)
    builder.add_tensor(1, DTYPE_F32, (n_floats,), USAGE_ACTIVATION)
    builder.add_tensor(2, DTYPE_F32, (n_floats,), USAGE_ACTIVATION)
    builder.add_tensor(3, DTYPE_F32, (n_floats,), USAGE_ACTIVATION)

    builder.add_node(0, "weighted_add", [0, 1], [2],
                     task_flags=TASK_REMOTE if 0 in remote_set else TASK_NONE)
    builder.add_node(1, "mock_relu", [2], [3],
                     task_flags=TASK_REMOTE if 1 in remote_set else TASK_NONE)
    builder.build(output)
    print(f"[nf_compiler] wrote {output}")


def _split_llama_mock(output: str, remote_nodes: Optional[str] = None):
    """Split-LLaMA mock: prefill(local) → relay(remote) → decode(remote).

    Tensor layout (LLaMA-like dimensions):
      T0: embed_weights   WEIGHT   [4096, 512]   — embedding table
      T1: input_tokens    ACTIV    [1, 128]       — token IDs
      T2: hidden_state    ACTIV    [1, 128, 4096] — prefill output
      T3: kv_cache        ACTIV    [1, 32, 128, 128] — KV cache
      T4: relay_out       ACTIV    [1, 128, 4096] — relayed hidden
      T5: logits          ACTIV    [1, 128, 32000] — output logits
    """
    embed_shape = (4096, 512)
    embed_weights = np.random.default_rng(42).standard_normal(
        embed_shape).astype(np.float32) * 0.02

    remote_set = set()
    if remote_nodes:
        remote_set = {int(x) for x in remote_nodes.split(',')}

    builder = NfirBuilder()
    # Tensors
    builder.add_tensor(0, DTYPE_F32, embed_shape, USAGE_WEIGHT,
                       weight_data=embed_weights)
    builder.add_tensor(1, DTYPE_I32, (1, 128), USAGE_ACTIVATION)
    builder.add_tensor(2, DTYPE_F32, (1, 128, 4096), USAGE_ACTIVATION)
    builder.add_tensor(3, DTYPE_F32, (1, 32, 128, 128), USAGE_ACTIVATION)
    builder.add_tensor(4, DTYPE_F32, (1, 128, 4096), USAGE_ACTIVATION)
    builder.add_tensor(5, DTYPE_F32, (1, 128, 32000), USAGE_ACTIVATION)

    # Nodes
    builder.add_node(0, "attention_prefill", [0, 1], [2, 3],
                     task_flags=TASK_REMOTE if 0 in remote_set else TASK_NONE)
    builder.add_node(1, "network_relay", [2, 3], [4],
                     task_flags=TASK_REMOTE)  # always remote
    builder.add_node(2, "decode_step", [4], [5],
                     task_flags=TASK_REMOTE if 2 in remote_set else TASK_NONE)
    builder.build(output)
    print(f"[nf_compiler] wrote {output} (split_llama_mock)")


PRESETS = {
    'demo': _demo_graph,
    'split_llama_mock': _split_llama_mock,
}


# ── GGUF → NFIR conversion ───────────────────────────────────────

# GGUF dtype → NF dtype mapping
_GGUF_TO_NF_DTYPE = {
    0: DTYPE_F32,   # GGUF_F32
    1: DTYPE_F16,   # GGUF_F16
    2: DTYPE_U8,    # GGUF_Q4_0 (raw quantized bytes)
    3: DTYPE_U8,    # GGUF_Q4_1
    6: DTYPE_U8,    # GGUF_Q5_0
    7: DTYPE_U8,    # GGUF_Q5_1
    8: DTYPE_U8,    # GGUF_Q8_0
    9: DTYPE_U8,    # GGUF_Q8_1
}


def _gguf_to_nfir(gguf_path: str, output: str):
    """Convert GGUF model weights to .nfir format."""
    from gguf_parser import GGUFParser

    with GGUFParser(gguf_path) as parser:
        tensors = parser.get_tensors()
        print(f"[nf_compiler] GGUF: {len(tensors)} tensors from {gguf_path}")

        builder = NfirBuilder()
        for i, t in enumerate(tensors):
            nf_dtype = _GGUF_TO_NF_DTYPE.get(t.dtype, DTYPE_U8)
            raw_data = parser.mm[t.data_offset:t.data_offset + t.data_size]
            # For quantized types, shape is flattened to byte count
            if nf_dtype == DTYPE_U8 and t.dtype not in (0, 1):
                shape = (t.data_size,)
            else:
                shape = t.shape
            builder.add_tensor(i, nf_dtype, shape, USAGE_WEIGHT,
                               weight_data=np.frombuffer(raw_data, dtype=np.uint8))

        # Default single-node identity graph (weights only, no compute)
        if tensors:
            builder.add_node(0, "identity", [0], [0])

        builder.build(output)
        print(f"[nf_compiler] wrote {output} (gguf, {len(tensors)} weight tensors)")


if __name__ == '__main__':
    output = '/tmp/demo_pipeline.nfir'
    remote = None
    preset = 'demo'
    gguf_path = None
    for arg in sys.argv[1:]:
        if arg.startswith('--output='):
            output = arg.split('=', 1)[1]
        elif arg.startswith('--remote-nodes='):
            remote = arg.split('=', 1)[1]
        elif arg.startswith('--preset='):
            preset = arg.split('=', 1)[1]
        elif arg.startswith('--gguf='):
            gguf_path = arg.split('=', 1)[1]
        elif arg == '--gguf':
            # Next arg is the path (handled below)
            pass

    # Handle --gguf <path> (space-separated)
    if '--gguf' in sys.argv and not gguf_path:
        idx = sys.argv.index('--gguf')
        if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith('--'):
            gguf_path = sys.argv[idx + 1]

    if gguf_path:
        _gguf_to_nfir(gguf_path, output)
    else:
        if preset not in PRESETS:
            print(f"Unknown preset '{preset}'. Available: {', '.join(PRESETS)}")
            sys.exit(1)
        PRESETS[preset](output, remote)
