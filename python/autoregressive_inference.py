#!/usr/bin/env python3
"""
autoregressive_inference.py — 10-step decode demo with TPS measurement

Usage:
    python3 autoregressive_inference.py <path-to-nfir>

Example:
    python3 ../tools/nf_compiler/export_nfir.py --preset split_llama_mock -o /tmp/split_llama_mock.nfir
    python3 autoregressive_inference.py /tmp/split_llama_mock.nfir
"""

import struct
import sys

from neurofabric import Engine, Session


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <nfir_path>", file=sys.stderr)
        sys.exit(1)

    nfir_path = sys.argv[1]
    n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    print(f"[nf] Loading {nfir_path}")
    engine = Engine(n_threads=4)
    session = Session(engine, nfir_path)

    print(f"[nf] Session: {session.num_nodes()} nodes, "
          f"{session.num_tensors()} tensors")

    # Set input tokens (tensor_id=1): SEQ_LEN int32 values
    seq_len = 128
    token_data = struct.pack(f"<{seq_len}i", *range(1, seq_len + 1))
    session.set_input(1, token_data)

    print(f"[nf] Running {n_steps} autoregressive decode steps...")
    total_us = 0.0
    for i in range(n_steps):
        session.step()
        step_us = session.last_step_us()
        total_us += step_us
        print(f"  step {i}: {step_us:.1f} us")

    tps = n_steps / (total_us * 1e-6) if total_us > 0 else 0
    print(f"\n[nf] {n_steps} decode steps: {total_us:.1f} us total, "
          f"{tps:.0f} TPS")


if __name__ == "__main__":
    main()
