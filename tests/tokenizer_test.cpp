/**
 * @file tokenizer_test.cpp
 * @brief Unit tests for BPE tokenizer
 *
 * Phase 24: Tokenizer Integration.
 *
 * Tests encode/decode with a synthetic vocabulary (no GGUF file needed),
 * plus optional real-model tests when NF_TEST_GGUF_PATH is set.
 */

#include "../tools/tokenizer.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "FAIL: %s @ %s:%d\n", #expr, __FILE__, __LINE__); \
        std::abort(); \
    } \
} while(0)

/* ================================================================== */
/*  Helper: build a minimal mock GGUFModel with a toy vocabulary       */
/* ================================================================== */
static nf::GGUFModel make_mock_model() {
    nf::GGUFModel m{};
    m.tokenizer_model = "llama";
    m.bos_id = 0;
    m.eos_id = 1;
    m.unk_id = 2;
    m.add_bos = true;
    m.add_eos = false;

    /* Toy vocab: BOS, EOS, UNK, space-prefixed words, individual chars, byte tokens */
    m.vocab = {
        "<s>",          /* 0: BOS */
        "</s>",         /* 1: EOS */
        "<unk>",        /* 2: UNK */
        "\xe2\x96\x81Hello",  /* 3: ▁Hello */
        "\xe2\x96\x81world",  /* 4: ▁world */
        "\xe2\x96\x81",       /* 5: ▁ (space) */
        "H",            /* 6 */
        "e",            /* 7 */
        "l",            /* 8 */
        "o",            /* 9 */
/* PLACEHOLDER_VOCAB_CONTINUE */
        "w",            /* 10 */
        "r",            /* 11 */
        "d",            /* 12 */
        "He",           /* 13 */
        "ll",           /* 14 */
        "lo",           /* 15 */
        "wo",           /* 16 */
        "rl",           /* 17 */
        "Hel",          /* 18 */
        "llo",          /* 19 */
        "wor",          /* 20 */
        "rld",          /* 21 */
        "Hell",         /* 22 */
        "ello",         /* 23 */
        "worl",         /* 24 */
        "Hello",        /* 25 */
        "world",        /* 26 */
        "<0x00>",       /* 27: byte 0 */
        "<0x01>",       /* 28: byte 1 */
        "<0xFF>",       /* 29: byte 255 */
    };

    m.vocab_size = static_cast<uint32_t>(m.vocab.size());

    /* Token types */
    m.token_types.resize(m.vocab_size, nf::TOKEN_NORMAL);
    m.token_types[0] = nf::TOKEN_CONTROL;  /* BOS */
    m.token_types[1] = nf::TOKEN_CONTROL;  /* EOS */
    m.token_types[2] = nf::TOKEN_UNKNOWN;  /* UNK */
    m.token_types[27] = nf::TOKEN_BYTE;
    m.token_types[28] = nf::TOKEN_BYTE;
    m.token_types[29] = nf::TOKEN_BYTE;

    /* Scores: higher = merge earlier in SPM mode */
    m.scores.resize(m.vocab_size, 0.0f);
    /* Give longer tokens higher scores so they merge first */
    for (uint32_t i = 0; i < m.vocab_size; ++i)
        m.scores[i] = static_cast<float>(m.vocab[i].size());
    /* Boost the full-word tokens */
    m.scores[3] = 100.0f;   /* ▁Hello */
    m.scores[4] = 100.0f;   /* ▁world */
    m.scores[25] = 50.0f;   /* Hello */
    m.scores[26] = 50.0f;   /* world */

    /* No explicit merges — use SPM score-based merging */
    m.merges.clear();

    return m;
}

/* ================================================================== */
/*  Test 1: Basic encode — known words                                 */
/* ================================================================== */
static void test_encode_basic() {
    auto model = make_mock_model();
    nf::Tokenizer tok(model);

    auto ids = tok.encode("Hello world");
    /* Expected: [BOS=0, ▁Hello=3, ▁world=4] */
    CHECK(ids.size() >= 2);
    CHECK(ids[0] == 0); /* BOS */

    /* The tokenizer should produce ▁Hello and ▁world */
    bool found_hello = false, found_world = false;
    for (auto id : ids) {
        if (id == 3) found_hello = true;
        if (id == 4) found_world = true;
    }
    CHECK(found_hello);
    CHECK(found_world);

    std::printf("  test_encode_basic: %zu tokens [", ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) std::printf(", ");
        std::printf("%d", ids[i]);
    }
    std::printf("] PASS\n");
}

/* ================================================================== */
/*  Test 2: Basic decode                                               */
/* ================================================================== */
static void test_decode_basic() {
    auto model = make_mock_model();
    nf::Tokenizer tok(model);

    std::vector<int32_t> ids = {0, 3, 4}; /* BOS, ▁Hello, ▁world */
    std::string text = tok.decode(ids);
    /* ▁Hello → " Hello", ▁world → " world" → " Hello world" */
    CHECK(text.find("Hello") != std::string::npos);
    CHECK(text.find("world") != std::string::npos);

    std::printf("  test_decode_basic: \"%s\" PASS\n", text.c_str());
}

/* ================================================================== */
/*  Test 3: Empty input                                                */
/* ================================================================== */
static void test_empty_input() {
    auto model = make_mock_model();
    nf::Tokenizer tok(model);

    auto ids = tok.encode("");
    /* Should just be [BOS] */
    CHECK(ids.size() == 1);
    CHECK(ids[0] == 0);

    std::printf("  test_empty_input: [BOS] only, PASS\n");
}

/* ================================================================== */
/*  Test 4: Special token IDs                                          */
/* ================================================================== */
static void test_special_tokens() {
    auto model = make_mock_model();
    nf::Tokenizer tok(model);

    CHECK(tok.bos_id() == 0);
    CHECK(tok.eos_id() == 1);
    CHECK(tok.vocab_size() == model.vocab_size);

    /* Decode should skip BOS and stop at EOS */
    std::vector<int32_t> ids = {0, 3, 1, 4}; /* BOS, ▁Hello, EOS, ▁world */
    std::string text = tok.decode(ids);
    CHECK(text.find("Hello") != std::string::npos);
    CHECK(text.find("world") == std::string::npos); /* stopped at EOS */

    std::printf("  test_special_tokens: PASS\n");
}

/* ================================================================== */
/*  Test 5: id_to_piece for byte tokens                                */
/* ================================================================== */
static void test_byte_tokens() {
    auto model = make_mock_model();
    nf::Tokenizer tok(model);

    /* Byte token <0x00> should decode to null byte */
    std::string p0 = tok.id_to_piece(27);
    CHECK(p0.size() == 1);
    CHECK(p0[0] == '\x00');

    /* Byte token <0xFF> should decode to 0xFF */
    std::string pff = tok.id_to_piece(29);
    CHECK(pff.size() == 1);
    CHECK(static_cast<unsigned char>(pff[0]) == 0xFF);

    std::printf("  test_byte_tokens: PASS\n");
}

/* ================================================================== */
/*  Test 6: Encode without BOS                                         */
/* ================================================================== */
static void test_no_bos() {
    auto model = make_mock_model();
    nf::Tokenizer tok(model);

    auto ids = tok.encode("Hello", false /* add_bos */);
    CHECK(!ids.empty());
    CHECK(ids[0] != 0); /* should NOT start with BOS */

    std::printf("  test_no_bos: %zu tokens, first=%d, PASS\n", ids.size(), ids[0]);
}

/* ================================================================== */
/*  Test 7: Real model tokenizer (optional, requires GGUF)             */
/* ================================================================== */
static void test_real_model_tokenizer() {
    const char* gguf_path = std::getenv("NF_TEST_GGUF_PATH");
    if (!gguf_path) {
        std::printf("  test_real_model_tokenizer: SKIPPED (no NF_TEST_GGUF_PATH)\n");
        return;
    }

    auto* model = nf::gguf_open(gguf_path);
    CHECK(model != nullptr);
    CHECK(!model->vocab.empty());
    CHECK(model->vocab_size > 0);

    nf::Tokenizer tok(*model);

    /* Encode a known string */
    auto ids = tok.encode("Hello, world!");
    CHECK(ids.size() >= 2); /* at least BOS + something */
    CHECK(ids[0] == static_cast<int32_t>(model->bos_id));

    /* Decode back */
    std::string decoded = tok.decode(ids);
    CHECK(decoded.find("Hello") != std::string::npos);
    CHECK(decoded.find("world") != std::string::npos);

    /* Round-trip: encode → decode should preserve content */
    std::string test_str = "The quick brown fox jumps over the lazy dog.";
    auto ids2 = tok.encode(test_str);
    std::string rt = tok.decode(ids2);
    /* Trim leading space (SentencePiece adds ▁ prefix) */
    if (!rt.empty() && rt[0] == ' ') rt = rt.substr(1);
    CHECK(rt == test_str);

    std::printf("  test_real_model_tokenizer: vocab=%u, encode(\"%s\")=%zu tokens\n",
                model->vocab_size, "Hello, world!", ids.size());
    std::printf("    round-trip: \"%s\" → %zu tokens → \"%s\" PASS\n",
                test_str.c_str(), ids2.size(), rt.c_str());

    nf::gguf_close(model);
}

/* ================================================================== */
/*  main                                                               */
/* ================================================================== */
int main() {
    std::printf("=== Tokenizer Tests ===\n");

    test_encode_basic();
    test_decode_basic();
    test_empty_input();
    test_special_tokens();
    test_byte_tokens();
    test_no_bos();
    test_real_model_tokenizer();

    std::printf("=== All tokenizer tests PASSED ===\n");
    return 0;
}
