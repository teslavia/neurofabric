/**
 * @file streaming_callback_test.cpp
 * @brief Phase 34-C: Streaming token callback API test
 */

#include "neuralOS/ddi/neuro_fabric_abi.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); return 1; } \
} while(0)

struct CallbackLog {
    std::vector<int32_t> tokens;
    std::vector<std::string> pieces;
};

static void test_callback(int32_t token_id, const char* piece, void* userdata) {
    auto* log = static_cast<CallbackLog*>(userdata);
    log->tokens.push_back(token_id);
    if (piece) log->pieces.push_back(piece);
}

int main() {
    std::printf("=== Phase 34-C: Streaming Callback Test ===\n");

    /* Test 1: Basic callback invocation */
    {
        nf_session_callbacks cb{};
        cb.on_token = test_callback;
        CallbackLog log;
        cb.userdata = &log;

        /* Simulate token stream */
        const char* pieces[] = {"Hello", " ", "world", "!"};
        for (int i = 0; i < 4; ++i) {
            if (cb.on_token) cb.on_token(i + 100, pieces[i], cb.userdata);
        }

        CHECK(log.tokens.size() == 4, "should receive 4 tokens");
        CHECK(log.tokens[0] == 100, "first token id");
        CHECK(log.tokens[3] == 103, "last token id");
        CHECK(log.pieces[0] == "Hello", "first piece");
        CHECK(log.pieces[3] == "!", "last piece");
        std::printf("  [PASS] basic callback invocation\n");
    }

    /* Test 2: Null callback is safe */
    {
        nf_session_callbacks cb{};
        cb.on_token = nullptr;
        cb.userdata = nullptr;
        /* Should not crash */
        if (cb.on_token) cb.on_token(0, "test", cb.userdata);
        std::printf("  [PASS] null callback safe\n");
    }

    /* Test 3: Callback with null piece */
    {
        CallbackLog log;
        nf_session_callbacks cb{};
        cb.on_token = test_callback;
        cb.userdata = &log;
        cb.on_token(42, nullptr, cb.userdata);
        CHECK(log.tokens.size() == 1, "received token with null piece");
        CHECK(log.tokens[0] == 42, "token id correct");
        CHECK(log.pieces.empty(), "no piece logged for null");
        std::printf("  [PASS] null piece handling\n");
    }

    /* Test 4: Token ordering preserved */
    {
        CallbackLog log;
        nf_session_callbacks cb{};
        cb.on_token = test_callback;
        cb.userdata = &log;
        for (int32_t i = 0; i < 100; ++i)
            cb.on_token(i, "x", cb.userdata);
        CHECK(log.tokens.size() == 100, "100 tokens received");
        for (int32_t i = 0; i < 100; ++i)
            CHECK(log.tokens[i] == i, "token ordering preserved");
        std::printf("  [PASS] token ordering (100 tokens)\n");
    }

    std::printf("=== All streaming callback tests passed ===\n");
    return 0;
}
