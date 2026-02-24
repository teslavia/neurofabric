/**
 * @file onnx_parser.hpp
 * @brief NeuralOS L1 — Zero-Dependency ONNX Protobuf Parser
 *
 * Phase 37.4: Hand-written protobuf decoder for ONNX ModelProto.
 * Only decodes the subset needed for transformer models (~40 ops).
 * No external protobuf dependency.
 */

#ifndef NEURALOS_ONNX_PARSER_HPP
#define NEURALOS_ONNX_PARSER_HPP

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>

namespace neuralOS { namespace onnx {

/* ================================================================== */
/*  Protobuf Wire Types                                                */
/* ================================================================== */

enum WireType : uint8_t {
    VARINT  = 0,
    FIXED64 = 1,
    LENGTH  = 2,
    FIXED32 = 5
};

/* ================================================================== */
/*  ONNX Data Structures (minimal subset)                              */
/* ================================================================== */

enum OnnxDataType : int32_t {
    ONNX_UNDEFINED = 0,
    ONNX_FLOAT     = 1,
    ONNX_UINT8     = 2,
    ONNX_INT8      = 3,
    ONNX_FLOAT16   = 10,
    ONNX_INT32     = 6,
    ONNX_INT64     = 7,
    ONNX_BFLOAT16  = 16
};

struct OnnxTensorType {
    OnnxDataType elem_type = ONNX_UNDEFINED;
    std::vector<int64_t> shape;
};

struct OnnxAttribute {
    std::string name;
    int64_t     i = 0;
    float       f = 0.0f;
    std::string s;
    std::vector<int64_t> ints;
    std::vector<float>   floats;
};

struct OnnxNode {
    std::string op_type;
    std::string name;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<OnnxAttribute> attributes;

    const OnnxAttribute* find_attr(const std::string& n) const {
        for (auto& a : attributes)
            if (a.name == n) return &a;
        return nullptr;
    }
};

struct OnnxTensor {
    std::string name;
    OnnxDataType data_type = ONNX_UNDEFINED;
    std::vector<int64_t> dims;
    std::vector<uint8_t> raw_data;
};

struct OnnxValueInfo {
    std::string name;
    OnnxTensorType type;
};

struct OnnxGraph {
    std::string name;
    std::vector<OnnxNode>      nodes;
    std::vector<OnnxTensor>    initializers;
    std::vector<OnnxValueInfo> inputs;
    std::vector<OnnxValueInfo> outputs;
};

struct OnnxModel {
    int64_t   ir_version    = 0;
    int64_t   model_version = 0;
    std::string producer_name;
    OnnxGraph   graph;
};

/* ================================================================== */
/*  ProtobufReader — minimal varint/length-delimited decoder           */
/* ================================================================== */

class ProtobufReader {
public:
    ProtobufReader(const uint8_t* data, size_t size)
        : data_(data), size_(size), pos_(0) {}

    bool at_end() const { return pos_ >= size_; }
    size_t pos() const { return pos_; }

    uint64_t read_varint() {
        uint64_t val = 0;
        int shift = 0;
        while (pos_ < size_) {
            uint8_t b = data_[pos_++];
            val |= static_cast<uint64_t>(b & 0x7F) << shift;
            if ((b & 0x80) == 0) break;
            shift += 7;
        }
        return val;
    }

    uint32_t read_tag() {
        return static_cast<uint32_t>(read_varint());
    }

    std::string read_string(size_t len) {
        std::string s(reinterpret_cast<const char*>(data_ + pos_), len);
        pos_ += len;
        return s;
    }

    std::vector<uint8_t> read_bytes(size_t len) {
        std::vector<uint8_t> v(data_ + pos_, data_ + pos_ + len);
        pos_ += len;
        return v;
    }

    void skip(size_t len) { pos_ += len; }

    uint32_t read_fixed32() {
        uint32_t v = 0;
        std::memcpy(&v, data_ + pos_, 4);
        pos_ += 4;
        return v;
    }

    uint64_t read_fixed64() {
        uint64_t v = 0;
        std::memcpy(&v, data_ + pos_, 8);
        pos_ += 8;
        return v;
    }

    float read_float() {
        float v = 0;
        std::memcpy(&v, data_ + pos_, 4);
        pos_ += 4;
        return v;
    }

    ProtobufReader sub_reader(size_t len) const {
        return ProtobufReader(data_ + pos_, len);
    }

    void skip_field(uint32_t wire_type) {
        switch (wire_type) {
            case VARINT:  read_varint(); break;
            case FIXED64: pos_ += 8; break;
            case LENGTH:  { auto l = read_varint(); pos_ += l; } break;
            case FIXED32: pos_ += 4; break;
            default: break;
        }
    }

private:
    const uint8_t* data_;
    size_t size_;
    size_t pos_;
};

/* ================================================================== */
/*  OnnxParser — parse binary ONNX protobuf into OnnxModel             */
/* ================================================================== */

class OnnxParser {
public:
    bool parse(const uint8_t* data, size_t size, OnnxModel* model) {
        if (!data || !model || size == 0) return false;
        ProtobufReader r(data, size);
        return parse_model(r, size, model);
    }

private:
    bool parse_model(ProtobufReader& r, size_t end, OnnxModel* m) {
        size_t start = r.pos();
        while (r.pos() - start < end && !r.at_end()) {
            uint32_t tag = r.read_tag();
            uint32_t field = tag >> 3;
            uint32_t wt = tag & 0x7;

            switch (field) {
                case 1: m->ir_version = static_cast<int64_t>(r.read_varint()); break;
                case 2: { /* producer_name */
                    auto len = r.read_varint();
                    m->producer_name = r.read_string(static_cast<size_t>(len));
                    break;
                }
                case 4: { /* model_version */
                    m->model_version = static_cast<int64_t>(r.read_varint());
                    break;
                }
                case 7: { /* graph */
                    auto len = r.read_varint();
                    auto sub = r.sub_reader(static_cast<size_t>(len));
                    parse_graph(sub, static_cast<size_t>(len), &m->graph);
                    r.skip(static_cast<size_t>(len));
                    break;
                }
                default: r.skip_field(wt); break;
            }
        }
        return true;
    }

    void parse_graph(ProtobufReader& r, size_t end, OnnxGraph* g) {
        size_t start = r.pos();
        while (r.pos() - start < end && !r.at_end()) {
            uint32_t tag = r.read_tag();
            uint32_t field = tag >> 3;
            uint32_t wt = tag & 0x7;

            switch (field) {
                case 1: { /* node */
                    auto len = r.read_varint();
                    auto sub = r.sub_reader(static_cast<size_t>(len));
                    OnnxNode node;
                    parse_node(sub, static_cast<size_t>(len), &node);
                    g->nodes.push_back(std::move(node));
                    r.skip(static_cast<size_t>(len));
                    break;
                }
                case 2: { /* name */
                    auto len = r.read_varint();
                    g->name = r.read_string(static_cast<size_t>(len));
                    break;
                }
                case 5: { /* initializer */
                    auto len = r.read_varint();
                    /* Skip initializer parsing for now (weights) */
                    r.skip(static_cast<size_t>(len));
                    break;
                }
                default: r.skip_field(wt); break;
            }
        }
    }

    void parse_node(ProtobufReader& r, size_t end, OnnxNode* n) {
        size_t start = r.pos();
        while (r.pos() - start < end && !r.at_end()) {
            uint32_t tag = r.read_tag();
            uint32_t field = tag >> 3;
            uint32_t wt = tag & 0x7;

            switch (field) {
                case 1: { /* input */
                    auto len = r.read_varint();
                    n->inputs.push_back(r.read_string(static_cast<size_t>(len)));
                    break;
                }
                case 2: { /* output */
                    auto len = r.read_varint();
                    n->outputs.push_back(r.read_string(static_cast<size_t>(len)));
                    break;
                }
                case 3: { /* name */
                    auto len = r.read_varint();
                    n->name = r.read_string(static_cast<size_t>(len));
                    break;
                }
                case 4: { /* op_type */
                    auto len = r.read_varint();
                    n->op_type = r.read_string(static_cast<size_t>(len));
                    break;
                }
                default: r.skip_field(wt); break;
            }
        }
    }
};

}} // namespace neuralOS::onnx

#endif // NEURALOS_ONNX_PARSER_HPP
