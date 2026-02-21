/**
 * @file TensorView.hpp
 * @brief Zero-overhead C++17 RAII wrapper over the C buffer ABI.
 *
 * INTERNAL TO CORE — never crosses a dynamic library boundary.
 * Provides:
 *   - RAII lifetime via retain/release (move-only, no copies)
 *   - Type-safe map/unmap via MappedSpan RAII guard
 *   - Zero-copy slicing for Context Hub cache reuse
 *   - Compile-time dtype dispatch
 *
 * This is the "upper C++ layer" of the hourglass:
 *   C++ TensorView (core-internal)
 *       |
 *   C ABI waist (neuro_buffer_abi.h)
 *       |
 *   C++ plugin internals (Metal/RKNN)
 */

#ifndef NF_TENSOR_VIEW_HPP
#define NF_TENSOR_VIEW_HPP

#include "neurofabric/neuro_buffer_abi.h"

#include <cassert>
#include <cstddef>
#include <cstring>
#include <span>
#include <array>
#include <utility>

namespace nf {

/* ------------------------------------------------------------------ */
/*  Compile-time dtype → C++ type mapping                              */
/* ------------------------------------------------------------------ */

template <nf_dtype D> struct dtype_traits;
template <> struct dtype_traits<NF_DTYPE_F32>  { using type = float;    };
template <> struct dtype_traits<NF_DTYPE_I32>  { using type = int32_t;  };
template <> struct dtype_traits<NF_DTYPE_I8>   { using type = int8_t;   };
template <> struct dtype_traits<NF_DTYPE_U8>   { using type = uint8_t;  };

template <nf_dtype D>
using dtype_t = typename dtype_traits<D>::type;

/* ------------------------------------------------------------------ */
/*  MappedSpan — RAII guard for mapped buffer access                   */
/*  Unmaps automatically on destruction. Move-only.                    */
/* ------------------------------------------------------------------ */

template <typename T>
class MappedSpan {
public:
    MappedSpan() = default;

    MappedSpan(nf_buffer buf, const nf_buffer_ops* ops,
               T* ptr, size_t count) noexcept
        : buf_(buf), ops_(ops), span_(ptr, count) {}

    ~MappedSpan() { reset(); }

    MappedSpan(MappedSpan&& o) noexcept
        : buf_(o.buf_), ops_(o.ops_), span_(o.span_) {
        o.buf_ = nullptr;
        o.ops_ = nullptr;
        o.span_ = {};
    }

    MappedSpan& operator=(MappedSpan&& o) noexcept {
        if (this != &o) {
            reset();
            buf_  = o.buf_;
            ops_  = o.ops_;
            span_ = o.span_;
            o.buf_ = nullptr;
            o.ops_ = nullptr;
            o.span_ = {};
        }
        return *this;
    }

    MappedSpan(const MappedSpan&) = delete;
    MappedSpan& operator=(const MappedSpan&) = delete;

    std::span<T> span() const noexcept { return span_; }
    T*           data() const noexcept { return span_.data(); }
    size_t       size() const noexcept { return span_.size(); }
    bool         valid() const noexcept { return buf_ != nullptr; }

    T& operator[](size_t i) const noexcept { return span_[i]; }

private:
    void reset() noexcept {
        if (buf_ && ops_ && ops_->unmap) {
            ops_->unmap(buf_);
        }
        buf_ = nullptr;
        ops_ = nullptr;
        span_ = {};
    }

    nf_buffer            buf_  = nullptr;
    const nf_buffer_ops* ops_  = nullptr;
    std::span<T>         span_ = {};
};

/* ------------------------------------------------------------------ */
/*  TensorView — RAII owner of an nf_buffer + its ops table            */
/*  Move-only. Ref-counted via the C ABI retain/release.               */
/* ------------------------------------------------------------------ */

class TensorView {
public:
    /* -- Construction / Destruction --------------------------------- */

    TensorView() = default;

    /** Take ownership of a buffer + ops pair (from plugin alloc). */
    TensorView(nf_buffer buf, nf_buffer_ops ops) noexcept
        : buf_(buf), ops_(ops) {}

    ~TensorView() { release(); }

    /* Move-only */
    TensorView(TensorView&& o) noexcept
        : buf_(o.buf_), ops_(o.ops_),
          slice_dim_(o.slice_dim_), slice_begin_(o.slice_begin_),
          slice_end_(o.slice_end_), slice_stride_(o.slice_stride_) {
        o.buf_ = nullptr;
        o.ops_ = {};
        o.slice_dim_ = 0;
        o.slice_begin_ = 0;
        o.slice_end_ = 0;
        o.slice_stride_ = 0;
    }

    TensorView& operator=(TensorView&& o) noexcept {
        if (this != &o) {
            release();
            buf_  = o.buf_;
            ops_  = o.ops_;
            slice_dim_    = o.slice_dim_;
            slice_begin_  = o.slice_begin_;
            slice_end_    = o.slice_end_;
            slice_stride_ = o.slice_stride_;
            o.buf_ = nullptr;
            o.ops_ = {};
            o.slice_dim_ = 0;
            o.slice_begin_ = 0;
            o.slice_end_ = 0;
            o.slice_stride_ = 0;
        }
        return *this;
    }

    TensorView(const TensorView&) = delete;
    TensorView& operator=(const TensorView&) = delete;

    /* -- Shared ownership (explicit retain) ------------------------- */

    /** Create a shared reference (increments refcount). */
    [[nodiscard]] TensorView share() const noexcept {
        if (buf_ && ops_.retain) {
            ops_.retain(buf_);
        }
        return TensorView(buf_, ops_);
    }

    /* -- Accessors -------------------------------------------------- */

    [[nodiscard]] bool valid() const noexcept { return buf_ != nullptr; }
    [[nodiscard]] nf_buffer handle() const noexcept { return buf_; }
    [[nodiscard]] const nf_buffer_ops& ops() const noexcept { return ops_; }

    [[nodiscard]] nf_buffer_info info() const noexcept {
        nf_buffer_info bi{};
        if (buf_ && ops_.get_info) {
            ops_.get_info(buf_, &bi);
        }
        return bi;
    }

    [[nodiscard]] nf_mem_domain domain() const noexcept {
        return info().domain;
    }

    [[nodiscard]] nf_tensor_desc desc() const noexcept {
        return info().desc;
    }

    /* -- Mapping ---------------------------------------------------- */

    /**
     * Map the buffer and return a typed RAII span.
     * Caller gets automatic unmap on scope exit.
     */
    template <typename T>
    [[nodiscard]] MappedSpan<T> map() const noexcept {
        if (!buf_ || !ops_.map) return {};
        void* ptr = nullptr;
        nf_status st = ops_.map(buf_, &ptr);
        if (st != NF_OK || !ptr) return {};
        auto bi = info();
        size_t count = bi.desc.size_bytes / sizeof(T);
        return MappedSpan<T>(buf_, &ops_, static_cast<T*>(ptr), count);
    }

    /* -- Cache Coherency -------------------------------------------- */

    nf_status flush(uint64_t offset = 0, uint64_t size = 0) const noexcept {
        if (!buf_ || !ops_.cache_sync) return NF_OK; // no-op if coherent
        return ops_.cache_sync(buf_, NF_CACHE_FLUSH, offset, size);
    }

    nf_status invalidate(uint64_t offset = 0, uint64_t size = 0) const noexcept {
        if (!buf_ || !ops_.cache_sync) return NF_OK;
        return ops_.cache_sync(buf_, NF_CACHE_INVALIDATE, offset, size);
    }

    nf_status flush_invalidate(uint64_t offset = 0, uint64_t size = 0) const noexcept {
        if (!buf_ || !ops_.cache_sync) return NF_OK;
        return ops_.cache_sync(buf_, NF_CACHE_FLUSH_INVALIDATE, offset, size);
    }

    /* -- Zero-Copy Export ------------------------------------------- */

    struct ExportToken {
        uint64_t      token  = 0;
        nf_mem_domain domain = NF_MEM_DOMAIN_CPU;
    };

    [[nodiscard]] ExportToken export_handle() const noexcept {
        ExportToken et{};
        if (buf_ && ops_.export_handle) {
            ops_.export_handle(buf_, &et.token, &et.domain);
        }
        return et;
    }

    /* -- Slicing (Zero-Copy Sub-View) ------------------------------- */

    /**
     * Create a sub-view into this tensor's memory.
     * The slice shares the same backing buffer (refcount incremented).
     * `dim` is the axis to slice along; [begin, end) is the range.
     *
     * This is the foundation for Context Hub cache reuse:
     * multiple agents can hold slices into the same KV-cache buffer
     * without any memory copy.
     */
    [[nodiscard]] TensorView slice(uint32_t dim,
                                   uint64_t begin,
                                   uint64_t end) const noexcept {
        if (!buf_ || !ops_.retain || !ops_.get_info) return {};

        nf_buffer_info bi = info();
        if (dim >= bi.desc.ndim) return {};
        if (begin >= end || end > bi.desc.shape[dim]) return {};

        // Compute byte offset for the slice start
        uint64_t stride = bi.desc.strides[dim];
        if (stride == 0) {
            // Contiguous: compute stride from inner dimensions
            stride = bi.desc.size_bytes;
            for (uint32_t d = 0; d < bi.desc.ndim; ++d) {
                if (d != dim) stride /= bi.desc.shape[d];
            }
        }

        // Build a new TensorView sharing the same buffer
        ops_.retain(buf_);
        TensorView sv(buf_, ops_);
        // The slice metadata is carried in a modified descriptor.
        // The actual offset is tracked via buffer_info.offset_bytes
        // in the plugin's internal bookkeeping. For Phase 2, we store
        // the slice parameters for the DAG scheduler to interpret.
        sv.slice_dim_    = dim;
        sv.slice_begin_  = begin;
        sv.slice_end_    = end;
        sv.slice_stride_ = stride;
        return sv;
    }

    /* Slice metadata accessors (for DAG scheduler) */
    [[nodiscard]] bool     is_slice() const noexcept { return slice_end_ > slice_begin_; }
    [[nodiscard]] uint32_t slice_dim() const noexcept { return slice_dim_; }
    [[nodiscard]] uint64_t slice_begin() const noexcept { return slice_begin_; }
    [[nodiscard]] uint64_t slice_end() const noexcept { return slice_end_; }
    [[nodiscard]] uint64_t slice_stride() const noexcept { return slice_stride_; }

private:
    void release() noexcept {
        if (buf_ && ops_.release) {
            ops_.release(buf_);
        }
        buf_ = nullptr;
        ops_ = {};
    }

    nf_buffer      buf_  = nullptr;
    nf_buffer_ops  ops_  = {};

    /* Slice metadata — zero for non-sliced views */
    uint32_t slice_dim_    = 0;
    uint64_t slice_begin_  = 0;
    uint64_t slice_end_    = 0;
    uint64_t slice_stride_ = 0;
};

} // namespace nf

#endif // NF_TENSOR_VIEW_HPP
