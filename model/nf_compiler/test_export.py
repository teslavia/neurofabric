"""
Unit tests for the Neuro-Fabric .nfir Python compiler.
Verifies binary layout matches neuro_ir_format.h exactly.
"""

import os
import struct
import tempfile
import unittest

import numpy as np

from export_nfir import (
    NfirBuilder, TensorDesc, NodeDesc, crc32c,
    DTYPE_F32, DTYPE_U8, USAGE_ACTIVATION, USAGE_WEIGHT,
    TASK_NONE, TASK_REMOTE,
    NF_IR_MAGIC, NF_IR_VERSION, NF_IR_PAYLOAD_ALIGN, NF_IR_WEIGHT_ALIGN,
    NF_MAX_DIMS, NF_MAX_OP_NAME, NF_IR_MAX_NODE_IO,
    HEADER_SIZE, TENSOR_DESC_SIZE, NODE_DESC_SIZE,
)


class TestCRC32C(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(crc32c(b''), 0x00000000)

    def test_known_vector(self):
        # CRC32C of "123456789" = 0xE3069283
        self.assertEqual(crc32c(b'123456789'), 0xE3069283)


class TestStructSizes(unittest.TestCase):
    def test_tensor_desc_size(self):
        td = TensorDesc(0, DTYPE_F32, (1024,), USAGE_WEIGHT)
        self.assertEqual(len(td.pack()), TENSOR_DESC_SIZE)

    def test_node_desc_size(self):
        nd = NodeDesc(0, "mock_relu", [0], [1])
        self.assertEqual(len(nd.pack()), NODE_DESC_SIZE)


class TestNfirBuilder(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, 'test.nfir')

    def tearDown(self):
        if os.path.exists(self.path):
            os.unlink(self.path)
        os.rmdir(self.tmpdir)

    def test_round_trip(self):
        n = 256
        weights = np.arange(n, dtype=np.float32) * 0.1

        builder = NfirBuilder()
        builder.add_tensor(0, DTYPE_F32, (n,), USAGE_WEIGHT, weight_data=weights)
        builder.add_tensor(1, DTYPE_F32, (n,), USAGE_ACTIVATION)
        builder.add_tensor(2, DTYPE_F32, (n,), USAGE_ACTIVATION)
        builder.add_node(0, "weighted_add", [0, 1], [2], task_flags=TASK_REMOTE)
        builder.build(self.path)

        with open(self.path, 'rb') as f:
            data = f.read()

        # ── Header validation ──
        hdr = data[:HEADER_SIZE]
        magic, version, num_t, num_n, pay_off, pay_sz, hdr_crc, pad = \
            struct.unpack('<IIIIQQ II', hdr)

        self.assertEqual(magic, NF_IR_MAGIC)
        self.assertEqual(version, NF_IR_VERSION)
        self.assertEqual(num_t, 3)
        self.assertEqual(num_n, 1)
        self.assertEqual(pay_off % NF_IR_PAYLOAD_ALIGN, 0)
        self.assertEqual(hdr_crc, crc32c(hdr[:32]))

        # ── Payload alignment ──
        self.assertGreaterEqual(len(data), pay_off + pay_sz)

        # ── Weight data at correct offset ──
        w_start = pay_off  # first weight at offset 0 within payload
        w_bytes = data[w_start:w_start + n * 4]
        recovered = np.frombuffer(w_bytes, dtype=np.float32)
        np.testing.assert_array_equal(recovered, weights)

        # ── Node task_flags ──
        node_offset = HEADER_SIZE + num_t * TENSOR_DESC_SIZE
        node_data = data[node_offset:node_offset + NODE_DESC_SIZE]
        # task_flags is at offset: 4 + 64 + 4 + 4 + 64 + 64 = 204
        flags = struct.unpack_from('<I', node_data, 204)[0]
        self.assertEqual(flags, TASK_REMOTE)

    def test_no_weights(self):
        builder = NfirBuilder()
        builder.add_tensor(0, DTYPE_F32, (64,), USAGE_ACTIVATION)
        builder.add_tensor(1, DTYPE_F32, (64,), USAGE_ACTIVATION)
        builder.add_node(0, "mock_relu", [0], [1])
        builder.build(self.path)

        with open(self.path, 'rb') as f:
            data = f.read()

        hdr = struct.unpack('<IIIIQQ II', data[:HEADER_SIZE])
        pay_sz = hdr[5]
        self.assertEqual(pay_sz, 0)


if __name__ == '__main__':
    unittest.main()
