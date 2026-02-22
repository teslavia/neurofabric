# NeuroFabric — Project Instructions

## Identity

Microkernel heterogeneous LLM inference engine. C++20 internally, C11 at ABI boundary.
Version: 0.1.1 | License: Apache 2.0 | Tests: 39/39 green

## Build

```bash
# macOS (Metal auto-detected on arm64)
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(sysctl -n hw.ncpu)

# RK3588 (native on board, IP: 192.168.1.70)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DNF_PLUGIN_RKNN=ON && cmake --build build -j$(nproc)

# Test
ctest --test-dir build --output-on-failure
```

## Architecture Rules

- **Zero-vptr ABI**: All cross-boundary calls use C POD structs + function pointers. No vtable, no RTTI, no exceptions across boundaries.
- **Plugin contract**: Each plugin exports exactly `nf_plugin_register` (+ optional `nf_plugin_register_mem`). Core exports 0 symbols.
- **ABI version gate**: `NF_ABI_VERSION` in `neuro_fabric_abi.h`. Bump patch for compatible changes, minor for additions, major for breaks.
- **GraphBuilder + mmap_buffer**: Compiled into test/CLI binaries directly, NOT exported from core library.
- **C/C++ dual headers**: Use `static_assert` in C++, `_Static_assert` in C. Guard with `__cplusplus`.

## Code Style

- C++20 standard, C11 at ABI boundary
- Symbol visibility hidden by default (`CMAKE_CXX_VISIBILITY_PRESET hidden`)
- Tests use `CHECK()` macro, never `assert()` (NDEBUG strips assert in Release)
- Metal `.mm` files compiled with `-fobjc-arc`
- MSL shaders embedded as `NSString` literals in metal_provider.mm
- Push constants at `[[buffer(15)]]` via `setBytes`

## Metal Kernel Conventions

- Op names: `"causal_attention"` (not `"causal_attn"`), `"metal_vector_add"` (not `"elementwise_add"`)
- argmax: `pc.N` = row width, `pc.seq_len` = row count (NOT `pc.head_dim`)
- linear_tiled: bounds check AFTER tile loading — all threads must hit `threadgroup_barrier`
- GQA: `pc.M` = n_kv_heads; `kv_head = head * n_kv / n_heads`; M==0 → standard MHA
- PSO registry: `MetalPSO` enum + `kPSOTable[]`, O(1) hash dispatch
- Init order: `g_vt.init(g_prov)` must precede any buffer allocation

## Key Directories

```
core/include/neurofabric/   ABI headers + C++ engine headers
core/src/                   Implementation + platform loaders
plugins/metal/src/          Metal GPU provider (53 kernels)
plugins/rknn/src/           RKNN NPU provider (DMA-BUF zero-copy)
plugins/network/src/        TCP proxy plugin
tools/                      CLI tools (nf_generate, nf_node_cli)
tools/model/                Header-only model libraries (GGUF, tokenizer, DAG builder, etc.)
tools/cross_compile/        Cross-compilation toolchain (scripts, Docker, board configs)
tools/nf_compiler/          Python AOT compiler
tests/                      39 test files
python/                     ctypes binding
```

## Git Workflow

- `main`: stable, release-tagged
- `dev`: active development
- Merge dev → main for releases, tag with `vX.Y.Z`
- Commit message format: `<scope>: <description>` (e.g., `metal: add flash_attention_paged kernel`)

## Common Tasks

```bash
# Run specific test
./build/bin/<test_name>

# Text generation
./build/bin/nf_generate model.gguf "prompt" --fp16 --paged --max-tokens 128

# Distributed mode
./build/bin/nf_node_cli --mode=worker --port=9999        # edge
./build/bin/nf_node_cli --mode=coord --nfir=model.nfir --remote=192.168.1.70:9999  # coordinator
```
