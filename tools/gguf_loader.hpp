/**
 * @file gguf_loader.hpp
 * @brief Header-only C++ GGUF v2/v3 mmap parser
 *
 * Phase 21: Real-Model GGUF→DAG End-to-End Inference.
 *
 * Opens a GGUF file via mmap, parses header + metadata KV pairs to extract
 * model hyperparameters, then walks tensor info entries to build a map of
 * tensor name → {dtype, shape, pointer into mmap'd region, byte_size}.
 *
 * Zero dependencies beyond POSIX (mmap) and C++ stdlib.
 */

#ifndef NF_GGUF_LOADER_HPP
#define NF_GGUF_LOADER_HPP

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace nf {

/* GGUF magic: "GGUF" as little-endian uint32 = 0x46554747 */
static constexpr uint32_t GGUF_MAGIC = 0x46554747;

/* GGUF metadata value types */
enum GGUFValueType : uint32_t {
    GGUF_TYPE_UINT8   = 0,  GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,  GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,  GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,  GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,  GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10, GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

/* GGUF tensor dtypes */
enum GGUFDtype : uint32_t {
    GGUF_DTYPE_F32  = 0, GGUF_DTYPE_F16  = 1,
    GGUF_DTYPE_Q4_0 = 2, GGUF_DTYPE_Q4_1 = 3,
/* PLACEHOLDER_CONTINUE */
    GGUF_DTYPE_Q5_0 = 6, GGUF_DTYPE_Q5_1 = 7,
    GGUF_DTYPE_Q8_0 = 8, GGUF_DTYPE_Q8_1 = 9,
    GGUF_DTYPE_Q2_K = 10, GGUF_DTYPE_Q3_K = 11,
    GGUF_DTYPE_Q4_K = 12, GGUF_DTYPE_Q5_K = 13,
    GGUF_DTYPE_Q6_K = 14,
};

/* Block size info: (bytes_per_block, elements_per_block) */
struct BlockInfo { uint32_t bytes; uint32_t elems; };

inline BlockInfo gguf_block_info(uint32_t dtype) {
    switch (dtype) {
        case GGUF_DTYPE_F32:  return {4, 1};
        case GGUF_DTYPE_F16:  return {2, 1};
        case GGUF_DTYPE_Q4_0: return {18, 32};
        case GGUF_DTYPE_Q4_1: return {20, 32};
        case GGUF_DTYPE_Q5_0: return {22, 32};
        case GGUF_DTYPE_Q5_1: return {24, 32};
        case GGUF_DTYPE_Q8_0: return {34, 32};
        case GGUF_DTYPE_Q8_1: return {36, 32};
        case GGUF_DTYPE_Q2_K: return {84, 256};
        case GGUF_DTYPE_Q3_K: return {110, 256};
        case GGUF_DTYPE_Q4_K: return {144, 256};
        case GGUF_DTYPE_Q5_K: return {176, 256};
        case GGUF_DTYPE_Q6_K: return {210, 256};
        default:              return {0, 0};
    }
}

/* Scalar sizes for metadata value types */
inline uint32_t gguf_scalar_size(uint32_t vtype) {
    switch (vtype) {
        case GGUF_TYPE_UINT8:  case GGUF_TYPE_INT8:  case GGUF_TYPE_BOOL: return 1;
        case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16: return 2;
        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: return 8;
        default: return 0;
    }
}

struct GGUFTensorInfo {
    uint32_t    gguf_dtype;
    uint32_t    ndim;
    uint64_t    shape[8];
    const void* data;       /* pointer into mmap'd region */
    size_t      byte_size;
    uint64_t    n_elements;
};

struct GGUFModel {
    /* mmap state */
    int         fd = -1;
    void*       mmap_base = MAP_FAILED;
    size_t      file_size = 0;

    /* Hyperparameters (from metadata KV pairs) */
    uint32_t dim = 0, n_layers = 0, n_heads = 0, n_kv_heads = 0;
    uint32_t ff_dim = 0, max_seq = 0, vocab_size = 0;
    float    rope_theta = 10000.0f;
    float    rms_norm_eps = 1e-5f;
    uint32_t alignment = 32;

    /* Tensor map: GGUF name → info with pointer into mmap */
    std::unordered_map<std::string, GGUFTensorInfo> tensors;
};

/* ---- Internal parsing helpers ---- */

namespace detail {

class GGUFReader {
public:
    GGUFReader(const uint8_t* base, size_t size)
        : base_(base), size_(size), pos_(0) {}

    template <typename T>
    T read() {
        T val;
        std::memcpy(&val, base_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        return val;
    }

    std::string read_string() {
        uint64_t len = read<uint64_t>();
        std::string s(reinterpret_cast<const char*>(base_ + pos_), len);
        pos_ += len;
        return s;
    }

    void skip_value(uint32_t vtype) {
        if (vtype == GGUF_TYPE_STRING) {
            read_string();
        } else if (vtype == GGUF_TYPE_ARRAY) {
            uint32_t elem_type = read<uint32_t>();
            uint64_t count = read<uint64_t>();
            for (uint64_t i = 0; i < count; ++i)
                skip_value(elem_type);
        } else {
            pos_ += gguf_scalar_size(vtype);
        }
    }

    uint64_t read_array_length() {
        /* reads elem_type (discards) + count */
        read<uint32_t>(); /* elem_type */
        return read<uint64_t>();
    }

    size_t pos() const { return pos_; }
    void set_pos(size_t p) { pos_ = p; }

private:
    const uint8_t* base_;
    size_t size_;
    size_t pos_;
};

} /* namespace detail */

inline uint64_t align_up(uint64_t val, uint64_t align) {
    return (val + align - 1) & ~(align - 1);
}

/**
 * Open a GGUF file, mmap it, parse header + metadata + tensor info.
 * Returns heap-allocated GGUFModel* on success, nullptr on failure.
 * Caller must call gguf_close() when done.
 */
inline GGUFModel* gguf_open(const char* path) {
    int fd = ::open(path, O_RDONLY);
    if (fd < 0) {
        std::fprintf(stderr, "[gguf_open] Cannot open: %s\n", path);
        return nullptr;
    }

    struct stat st;
    if (::fstat(fd, &st) != 0) {
        ::close(fd);
        return nullptr;
    }
    size_t file_size = static_cast<size_t>(st.st_size);

    void* base = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) {
        ::close(fd);
        return nullptr;
    }

    auto* m = new GGUFModel;
    m->fd = fd;
    m->mmap_base = base;
    m->file_size = file_size;

    detail::GGUFReader r(reinterpret_cast<const uint8_t*>(base), file_size);

    /* 1. Header: magic, version, tensor_count, kv_count */
    uint32_t magic = r.read<uint32_t>();
    uint32_t version = r.read<uint32_t>();
    if (magic != GGUF_MAGIC || (version != 2 && version != 3)) {
        std::fprintf(stderr, "[gguf_open] Bad magic/version: 0x%08X v%u\n",
                     magic, version);
        ::munmap(base, file_size);
        ::close(fd);
        delete m;
        return nullptr;
    }
    uint64_t tensor_count = r.read<uint64_t>();
    uint64_t kv_count = r.read<uint64_t>();

    /* 2. Walk metadata KV pairs — extract known hyperparameters */
    for (uint64_t i = 0; i < kv_count; ++i) {
        std::string key = r.read_string();
        uint32_t vtype = r.read<uint32_t>();

        if (key == "llama.embedding_length" && vtype == GGUF_TYPE_UINT32) {
            m->dim = r.read<uint32_t>();
        } else if (key == "llama.block_count" && vtype == GGUF_TYPE_UINT32) {
            m->n_layers = r.read<uint32_t>();
        } else if (key == "llama.attention.head_count" && vtype == GGUF_TYPE_UINT32) {
            m->n_heads = r.read<uint32_t>();
        } else if (key == "llama.attention.head_count_kv" && vtype == GGUF_TYPE_UINT32) {
            m->n_kv_heads = r.read<uint32_t>();
        } else if (key == "llama.feed_forward_length" && vtype == GGUF_TYPE_UINT32) {
            m->ff_dim = r.read<uint32_t>();
        } else if (key == "llama.context_length" && vtype == GGUF_TYPE_UINT32) {
            m->max_seq = r.read<uint32_t>();
        } else if (key == "llama.rope.freq_base" && vtype == GGUF_TYPE_FLOAT32) {
            m->rope_theta = r.read<float>();
        } else if (key == "llama.attention.layer_norm_rms_epsilon" && vtype == GGUF_TYPE_FLOAT32) {
            m->rms_norm_eps = r.read<float>();
        } else if (key == "general.alignment" && vtype == GGUF_TYPE_UINT32) {
            m->alignment = r.read<uint32_t>();
        } else if (key == "tokenizer.ggml.tokens" && vtype == GGUF_TYPE_ARRAY) {
            /* Array count gives vocab_size */
            size_t saved = r.pos();
            m->vocab_size = static_cast<uint32_t>(r.read_array_length());
            r.set_pos(saved);
            r.skip_value(vtype);
        } else {
            r.skip_value(vtype);
        }
    }

    /* 3. Walk tensor info entries */
    struct RawTensorInfo {
        std::string name;
        uint32_t ndim;
        uint64_t shape[8];
        uint32_t dtype;
        uint64_t offset;
        size_t byte_size;
        uint64_t n_elements;
    };
    std::vector<RawTensorInfo> raw_tensors;
    raw_tensors.reserve(tensor_count);

    for (uint64_t i = 0; i < tensor_count; ++i) {
        RawTensorInfo ti{};
        ti.name = r.read_string();
        ti.ndim = r.read<uint32_t>();
        for (uint32_t d = 0; d < ti.ndim; ++d)
            ti.shape[d] = r.read<uint64_t>();
        ti.dtype = r.read<uint32_t>();
        ti.offset = r.read<uint64_t>();

        /* Compute n_elements and byte_size */
        ti.n_elements = 1;
        for (uint32_t d = 0; d < ti.ndim; ++d)
            ti.n_elements *= ti.shape[d];

        BlockInfo bi = gguf_block_info(ti.dtype);
        if (bi.elems > 0) {
            uint64_t n_blocks = (ti.n_elements + bi.elems - 1) / bi.elems;
            ti.byte_size = static_cast<size_t>(n_blocks * bi.bytes);
        } else {
            ti.byte_size = 0;
        }
        raw_tensors.push_back(std::move(ti));
    }

    /* 4. Compute data_start = align_up(cursor, alignment) */
    uint64_t data_start = align_up(r.pos(), m->alignment);

    /* 5. Build tensor map with pointers into mmap */
    auto* base_u8 = reinterpret_cast<const uint8_t*>(base);
    for (auto& ti : raw_tensors) {
        GGUFTensorInfo info{};
        info.gguf_dtype = ti.dtype;
        info.ndim = ti.ndim;
        for (uint32_t d = 0; d < ti.ndim; ++d)
            info.shape[d] = ti.shape[d];
        info.data = base_u8 + data_start + ti.offset;
        info.byte_size = ti.byte_size;
        info.n_elements = ti.n_elements;
        m->tensors[ti.name] = info;
    }

    return m;
}

inline void gguf_close(GGUFModel* m) {
    if (!m) return;
    if (m->mmap_base != MAP_FAILED)
        ::munmap(m->mmap_base, m->file_size);
    if (m->fd >= 0)
        ::close(m->fd);
    delete m;
}

} /* namespace nf */

#endif /* NF_GGUF_LOADER_HPP */
