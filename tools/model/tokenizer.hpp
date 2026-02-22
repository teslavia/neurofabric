/**
 * @file tokenizer.hpp
 * @brief Header-only BPE tokenizer for LLaMA / SentencePiece models
 *
 * Phase 24: Tokenizer Integration.
 *
 * Reads vocab, merges, scores, and special tokens from a parsed GGUFModel.
 * Supports both SentencePiece (LLaMA) and BPE (GPT-2 style) tokenization.
 * Zero external dependencies.
 */

#ifndef NF_TOKENIZER_HPP
#define NF_TOKENIZER_HPP

#include "gguf_loader.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <queue>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nf {

/* Token type constants matching GGUF spec */
enum TokenType : int32_t {
    TOKEN_NORMAL      = 1,
    TOKEN_UNKNOWN     = 2,
    TOKEN_CONTROL     = 3,
    TOKEN_USER_DEF    = 4,
    TOKEN_UNUSED      = 5,
    TOKEN_BYTE        = 6,
};

class Tokenizer {
public:
    explicit Tokenizer(const GGUFModel& model) {
        vocab_size_ = model.vocab_size;
        bos_id_ = model.bos_id;
        eos_id_ = model.eos_id;
        unk_id_ = model.unk_id;
        add_bos_ = model.add_bos;
/* PLACEHOLDER_CTOR_CONTINUE */
        add_eos_ = model.add_eos;
        is_spm_ = (model.tokenizer_model == "llama");

        /* Build id→token and token→id maps */
        id_to_token_.resize(vocab_size_);
        for (uint32_t i = 0; i < vocab_size_ && i < model.vocab.size(); ++i) {
            id_to_token_[i] = model.vocab[i];
            token_to_id_[model.vocab[i]] = static_cast<int32_t>(i);
        }

        /* Copy scores */
        scores_ = model.scores;
        scores_.resize(vocab_size_, 0.0f);

        /* Copy token types */
        token_types_ = model.token_types;
        token_types_.resize(vocab_size_, TOKEN_NORMAL);

        /* Build byte token lookup: <0x00> .. <0xFF> */
        for (uint32_t i = 0; i < vocab_size_; ++i) {
            const auto& tok = id_to_token_[i];
            if (tok.size() == 6 && tok[0] == '<' && tok[1] == '0'
                && tok[2] == 'x' && tok[5] == '>') {
                unsigned val = 0;
                if (std::sscanf(tok.c_str(), "<0x%02X>", &val) == 1 && val < 256)
                    byte_to_id_[val] = static_cast<int32_t>(i);
            }
        }

        /* Build BPE merge ranks from merges list */
        for (size_t i = 0; i < model.merges.size(); ++i) {
            merge_ranks_[model.merges[i]] = static_cast<int32_t>(i);
        }
    }

    /* ---- Encode: text → token IDs ---- */
    std::vector<int32_t> encode(const std::string& text, bool add_bos = true) const {
        std::vector<int32_t> tokens;
        if (add_bos && add_bos_)
            tokens.push_back(static_cast<int32_t>(bos_id_));

        if (text.empty()) return tokens;

        if (is_spm_) {
            /* SentencePiece: prepend space, split into chars, then BPE merge */
            std::string input = "\xe2\x96\x81" + text; /* ▁ + text */
            /* Replace spaces with ▁ */
            std::string processed;
            processed.reserve(input.size() * 2);
            for (size_t i = 0; i < input.size(); ++i) {
                if (input[i] == ' ') {
                    processed += "\xe2\x96\x81"; /* ▁ (U+2581) */
                } else {
                    processed += input[i];
                }
            }
            encode_bpe(processed, tokens);
        } else {
            /* GPT-2 style: split on whitespace, BPE each word */
            encode_bpe(text, tokens);
        }

        if (add_eos_)
            tokens.push_back(static_cast<int32_t>(eos_id_));

        return tokens;
    }

    /* ---- Decode: token IDs → text ---- */
    std::string decode(const std::vector<int32_t>& tokens) const {
        std::string result;
        for (auto id : tokens) {
            if (id == static_cast<int32_t>(bos_id_)) continue;
            if (id == static_cast<int32_t>(eos_id_)) break;
            result += id_to_piece(id);
        }
        return result;
    }

    /* ---- Single token → string piece ---- */
    std::string id_to_piece(int32_t id) const {
        if (id < 0 || static_cast<uint32_t>(id) >= vocab_size_)
            return "";
        const std::string& tok = id_to_token_[id];

        /* Byte token: <0xNN> → raw byte */
        if (token_types_.size() > static_cast<size_t>(id)
            && token_types_[id] == TOKEN_BYTE) {
            unsigned val = 0;
            if (std::sscanf(tok.c_str(), "<0x%02X>", &val) == 1)
                return std::string(1, static_cast<char>(val));
        }

        /* SentencePiece: ▁ → space */
        if (is_spm_) {
            std::string piece;
            piece.reserve(tok.size());
            for (size_t i = 0; i < tok.size(); ) {
                if (i + 2 < tok.size()
                    && tok[i] == '\xe2' && tok[i+1] == '\x96' && tok[i+2] == '\x81') {
                    piece += ' ';
                    i += 3;
                } else {
                    piece += tok[i];
                    ++i;
                }
            }
            return piece;
        }
        return tok;
    }

    uint32_t bos_id() const { return bos_id_; }
    uint32_t eos_id() const { return eos_id_; }
    uint32_t vocab_size() const { return vocab_size_; }
/* PLACEHOLDER_PRIVATE */

private:
    /* BPE encode: split text into initial tokens, then iteratively merge */
    void encode_bpe(const std::string& text, std::vector<int32_t>& out) const {
        /* Split into UTF-8 characters as initial symbols */
        std::vector<std::string> symbols;
        for (size_t i = 0; i < text.size(); ) {
            size_t char_len = utf8_len(text[i]);
            if (i + char_len > text.size()) char_len = 1;
            symbols.push_back(text.substr(i, char_len));
            i += char_len;
        }

        /* Try to match each symbol to a vocab token; if not found, use byte fallback */
        std::vector<std::string> pieces;
        for (auto& sym : symbols) {
            if (token_to_id_.count(sym)) {
                pieces.push_back(sym);
            } else {
                /* Byte fallback: each byte becomes a separate piece */
                for (unsigned char c : sym)
                    pieces.push_back(byte_piece(c));
            }
        }

        /* O(n log n) BPE merge using doubly-linked list + priority queue */
        bool use_merges = !merge_ranks_.empty();

        struct Symbol { std::string text; int prev; int next; size_t gen; };
        struct Merge {
            float priority;   /* BPE: rank (lower=better), SPM: -score (lower=better) */
            int   left;
            int   right;
            size_t gen;       /* generation of left symbol when this merge was pushed */
        };
        auto merge_greater = [](const Merge& a, const Merge& b) {
            return a.priority > b.priority;
        };

        const int n = static_cast<int>(pieces.size());
        std::vector<Symbol> syms(n);
        for (int i = 0; i < n; ++i) {
            syms[i].text = std::move(pieces[i]);
            syms[i].prev = i - 1;
            syms[i].next = (i + 1 < n) ? i + 1 : -1;
            syms[i].gen  = 0;
        }

        /* Try to score an adjacent pair; returns {priority, true} or {0, false} */
        auto try_pair = [&](int li, int ri) -> std::pair<float, bool> {
            if (li < 0 || ri < 0) return {0.0f, false};
            if (use_merges) {
                std::string pair = syms[li].text + " " + syms[ri].text;
                auto it = merge_ranks_.find(pair);
                if (it != merge_ranks_.end())
                    return {static_cast<float>(it->second), true};
            } else if (!scores_.empty()) {
                std::string merged = syms[li].text + syms[ri].text;
                auto it = token_to_id_.find(merged);
                if (it != token_to_id_.end())
                    return {-scores_[it->second], true};  /* negate: higher score = lower priority */
            }
            return {0.0f, false};
        };

        std::priority_queue<Merge, std::vector<Merge>, decltype(merge_greater)> pq(merge_greater);

        /* Seed the heap with all initial adjacent pairs */
        for (int i = 0; i + 1 < n; ++i) {
            auto [pri, ok] = try_pair(i, i + 1);
            if (ok) pq.push({pri, i, i + 1, syms[i].gen});
        }

        /* Pop loop */
        while (!pq.empty()) {
            auto [pri, li, ri, gen] = pq.top();
            pq.pop();

            /* Stale check: skip if left was re-merged, right was consumed, or either is tombstoned */
            if (gen != syms[li].gen) continue;              /* left was re-merged */
            if (syms[li].next != ri) continue;              /* right was consumed */
            if (syms[ri].prev == -1 && ri != 0) continue;   /* right is tombstoned */
            if (syms[li].prev == -1 && li != 0) continue;   /* left is tombstoned */

            /* Merge: absorb right's text into left */
            syms[li].text += syms[ri].text;
            syms[li].gen++;

            /* Unlink right from the linked list */
            int rn = syms[ri].next;
            syms[li].next = rn;
            if (rn >= 0) syms[rn].prev = li;
            syms[ri].prev = -1;  /* tombstone */

            /* Push new pairs involving the merged symbol */
            int lp = syms[li].prev;
            if (lp >= 0) {
                auto [p2, ok2] = try_pair(lp, li);
                if (ok2) pq.push({p2, lp, li, syms[lp].gen});
            }
            if (rn >= 0) {
                auto [p3, ok3] = try_pair(li, rn);
                if (ok3) pq.push({p3, li, rn, syms[li].gen});
            }
        }

        /* Collect surviving symbols back into pieces */
        pieces.clear();
        for (int i = 0; i >= 0; i = syms[i].next)
            pieces.push_back(syms[i].text);

        /* Map pieces to token IDs */
        for (auto& p : pieces) {
            auto it = token_to_id_.find(p);
            if (it != token_to_id_.end()) {
                out.push_back(it->second);
            } else {
                out.push_back(static_cast<int32_t>(unk_id_));
            }
        }
    }

    /* Get byte fallback piece string */
    std::string byte_piece(unsigned char c) const {
        auto it = byte_to_id_.find(c);
        if (it != byte_to_id_.end())
            return id_to_token_[it->second];
        /* Fallback: construct <0xNN> */
        char buf[8];
        std::snprintf(buf, sizeof(buf), "<0x%02X>", c);
        return buf;
    }

    static size_t utf8_len(unsigned char c) {
        if ((c & 0x80) == 0)    return 1;
        if ((c & 0xE0) == 0xC0) return 2;
        if ((c & 0xF0) == 0xE0) return 3;
        if ((c & 0xF8) == 0xF0) return 4;
        return 1;
    }

    /* Member data */
    uint32_t vocab_size_ = 0;
    uint32_t bos_id_ = 1, eos_id_ = 2, unk_id_ = 0;
    bool add_bos_ = true, add_eos_ = false;
    bool is_spm_ = false;

    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<std::string, int32_t> merge_ranks_;
    std::vector<float> scores_;
    std::vector<int32_t> token_types_;
    std::unordered_map<unsigned char, int32_t> byte_to_id_;
};

} /* namespace nf */

#endif /* NF_TOKENIZER_HPP */