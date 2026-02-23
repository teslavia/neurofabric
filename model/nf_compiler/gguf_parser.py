"""
GGUF v2/v3 Parser — mmap-based, zero dependencies beyond stdlib.

Parses GGUF header, skips metadata KV pairs, extracts tensor info
(name, shape, dtype, data offset/size) for downstream .nfir conversion.

Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
"""

import mmap
import struct
import sys
from dataclasses import dataclass
from typing import List

# GGUF magic: "GGUF" in little-endian
GGUF_MAGIC = 0x46475547

# GGUF metadata value types
GGUF_TYPE_UINT8    = 0
GGUF_TYPE_INT8     = 1
GGUF_TYPE_UINT16   = 2
GGUF_TYPE_INT16    = 3
GGUF_TYPE_UINT32   = 4
GGUF_TYPE_INT32    = 5
GGUF_TYPE_FLOAT32  = 6
GGUF_TYPE_BOOL     = 7
GGUF_TYPE_STRING   = 8
GGUF_TYPE_ARRAY    = 9
GGUF_TYPE_UINT64   = 10
GGUF_TYPE_INT64    = 11
GGUF_TYPE_FLOAT64  = 12

_SCALAR_SIZES = {
    GGUF_TYPE_UINT8: 1, GGUF_TYPE_INT8: 1,
    GGUF_TYPE_UINT16: 2, GGUF_TYPE_INT16: 2,
    GGUF_TYPE_UINT32: 4, GGUF_TYPE_INT32: 4,
    GGUF_TYPE_FLOAT32: 4, GGUF_TYPE_BOOL: 1,
    GGUF_TYPE_UINT64: 8, GGUF_TYPE_INT64: 8,
    GGUF_TYPE_FLOAT64: 8,
}

# GGUF tensor dtypes
GGUF_DTYPE_F32  = 0
GGUF_DTYPE_F16  = 1
GGUF_DTYPE_Q4_0 = 2
GGUF_DTYPE_Q4_1 = 3
GGUF_DTYPE_Q5_0 = 6
GGUF_DTYPE_Q5_1 = 7
GGUF_DTYPE_Q8_0 = 8
GGUF_DTYPE_Q8_1 = 9

# Bytes per element (for quantized: bytes per block / elements per block)
_DTYPE_BLOCK_SIZE = {
    GGUF_DTYPE_F32: (4, 1), GGUF_DTYPE_F16: (2, 1),
    GGUF_DTYPE_Q4_0: (18, 32), GGUF_DTYPE_Q4_1: (20, 32),
    GGUF_DTYPE_Q5_0: (22, 32), GGUF_DTYPE_Q5_1: (24, 32),
    GGUF_DTYPE_Q8_0: (34, 32), GGUF_DTYPE_Q8_1: (36, 32),
}

GGUF_DTYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
    6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
}

# Default alignment for tensor data
GGUF_DEFAULT_ALIGNMENT = 32


@dataclass
class GGUFTensor:
    name: str
    ndim: int
    shape: tuple
    dtype: int
    data_offset: int   # absolute offset in file
    data_size: int     # byte size of tensor data


class GGUFParser:
    """Parse GGUF v2/v3 files via mmap. Read-only, zero-copy."""

    def __init__(self, path: str):
        self.path = path
        self.fd = open(path, 'rb')
        self.mm = mmap.mmap(self.fd.fileno(), 0, access=mmap.ACCESS_READ)
        self.version = 0
        self.tensor_count = 0
        self.kv_count = 0
        self.alignment = GGUF_DEFAULT_ALIGNMENT
        self._pos = 0
        self._tensors: List[GGUFTensor] = []
        self._metadata: dict = {}

        self._parse_header()
        self._skip_metadata()
        self._parse_tensor_info()

    def close(self):
        self.mm.close()
        self.fd.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _read(self, fmt: str) -> tuple:
        size = struct.calcsize(fmt)
        vals = struct.unpack_from(fmt, self.mm, self._pos)
        self._pos += size
        return vals

    def _read_string(self) -> str:
        (length,) = self._read('<Q')
        s = self.mm[self._pos:self._pos + length].decode('utf-8', errors='replace')
        self._pos += length
        return s

    def _skip_value(self, vtype: int):
        if vtype == GGUF_TYPE_STRING:
            self._read_string()
        elif vtype == GGUF_TYPE_ARRAY:
            (elem_type,) = self._read('<I')
            (count,) = self._read('<Q')
            for _ in range(count):
                self._skip_value(elem_type)
        elif vtype in _SCALAR_SIZES:
            self._pos += _SCALAR_SIZES[vtype]
        else:
            raise ValueError(f"Unknown GGUF value type: {vtype}")

    def _parse_header(self):
        magic, version = self._read('<II')
        assert magic == GGUF_MAGIC, f"Not a GGUF file (magic=0x{magic:08X})"
        assert version in (2, 3), f"Unsupported GGUF version {version}"
        self.version = version
        (self.tensor_count,) = self._read('<Q')
        (self.kv_count,) = self._read('<Q')

    def _skip_metadata(self):
        for _ in range(self.kv_count):
            key = self._read_string()
            (vtype,) = self._read('<I')
            # Capture alignment if present
            if key == "general.alignment" and vtype == GGUF_TYPE_UINT32:
                (self.alignment,) = self._read('<I')
            else:
                self._skip_value(vtype)
            self._metadata[key] = None  # track keys only

    def _parse_tensor_info(self):
        infos = []
        for _ in range(self.tensor_count):
            name = self._read_string()
            (ndim,) = self._read('<I')
            shape = tuple(self._read(f'<{ndim}Q'))
            (dtype,) = self._read('<I')
            (offset,) = self._read('<Q')

            # Compute data size
            n_elements = 1
            for d in shape:
                n_elements *= d
            block_bytes, block_elems = _DTYPE_BLOCK_SIZE.get(dtype, (1, 1))
            n_blocks = (n_elements + block_elems - 1) // block_elems
            data_size = n_blocks * block_bytes

            infos.append((name, ndim, shape, dtype, offset, data_size))

        # Tensor data starts after all metadata + tensor info, aligned
        data_start = self._align_up(self._pos, self.alignment)

        for name, ndim, shape, dtype, offset, data_size in infos:
            self._tensors.append(GGUFTensor(
                name=name, ndim=ndim, shape=shape, dtype=dtype,
                data_offset=data_start + offset,
                data_size=data_size,
            ))

    @staticmethod
    def _align_up(val: int, align: int) -> int:
        return (val + align - 1) & ~(align - 1)

    def get_tensors(self) -> List[GGUFTensor]:
        return list(self._tensors)

    def get_metadata_keys(self) -> List[str]:
        return list(self._metadata.keys())

    def summary(self) -> str:
        lines = [
            f"GGUF v{self.version}: {self.tensor_count} tensors, "
            f"{self.kv_count} metadata keys, alignment={self.alignment}",
        ]
        for t in self._tensors:
            dtype_name = GGUF_DTYPE_NAMES.get(t.dtype, f"?{t.dtype}")
            shape_str = "x".join(str(d) for d in t.shape)
            lines.append(
                f"  {t.name:40s}  {dtype_name:6s}  [{shape_str}]  "
                f"{t.data_size:>12,} bytes @ 0x{t.data_offset:X}"
            )
        return "\n".join(lines)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} [--info] <file.gguf>")
        sys.exit(1)

    path = sys.argv[-1]
    with GGUFParser(path) as parser:
        print(parser.summary())
