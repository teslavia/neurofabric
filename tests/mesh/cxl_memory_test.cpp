/**
 * @file cxl_memory_test.cpp
 * @brief Phase 38.4 — CXL memory domain abstraction tests
 */

#include "neuralOS/mesh/cxl_memory.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg); \
        exit(1); \
    } \
} while(0)

int main() {
    printf("=== CXL Memory Test ===\n");

    neuralOS::mesh::CxlMemoryPool::Config cfg;
    cfg.pool_size_bytes = 1024 * 1024;  /* 1 MB for testing */
    cfg.simulated_latency_ns = 0;       /* disable latency for speed */
    cfg.shared = false;

    neuralOS::mesh::CxlMemoryPool pool(cfg);

    /* Test 1: Alloc */
    uint64_t a1 = pool.cxl_alloc(4096);
    CHECK(a1 > 0, "alloc 4KB");
    CHECK(pool.used_bytes() == 4096, "used 4KB");
    CHECK(pool.num_allocs() == 1, "1 allocation");

    uint64_t a2 = pool.cxl_alloc(8192);
    CHECK(a2 > 0, "alloc 8KB");
    CHECK(pool.used_bytes() == 4096 + 8192, "used 12KB");

    /* Test 2: Store and load */
    uint8_t write_buf[256];
    std::memset(write_buf, 0xAA, 256);
    CHECK(pool.cxl_store(a1, 0, write_buf, 256), "store OK");

    uint8_t read_buf[256] = {};
    CHECK(pool.cxl_load(a1, 0, read_buf, 256), "load OK");
    CHECK(read_buf[0] == 0xAA && read_buf[255] == 0xAA, "data correct");

    /* Test 3: Store at offset */
    uint8_t pattern = 0x55;
    CHECK(pool.cxl_store(a1, 100, &pattern, 1), "store at offset");
    uint8_t loaded = 0;
    CHECK(pool.cxl_load(a1, 100, &loaded, 1), "load at offset");
    CHECK(loaded == 0x55, "offset data correct");

    /* Test 4: Out of bounds */
    CHECK(!pool.cxl_store(a1, 4000, write_buf, 256), "OOB store fails");
    CHECK(!pool.cxl_load(a1, 4000, read_buf, 256), "OOB load fails");

    /* Test 5: Invalid alloc_id */
    CHECK(!pool.cxl_store(9999, 0, write_buf, 1), "invalid id store fails");
    CHECK(!pool.cxl_load(9999, 0, read_buf, 1), "invalid id load fails");

    /* Test 6: Free */
    CHECK(pool.cxl_free(a1), "free a1");
    CHECK(pool.used_bytes() == 8192, "used 8KB after free");
    CHECK(pool.num_allocs() == 1, "1 allocation after free");
    CHECK(!pool.cxl_free(a1), "double free fails");

    /* Test 7: OOM */
    uint64_t big = pool.cxl_alloc(2 * 1024 * 1024);
    CHECK(big == 0, "OOM returns 0");

    /* Test 8: Shared pool */
    neuralOS::mesh::CxlMemoryPool::Config shared_cfg;
    shared_cfg.pool_size_bytes = 4096;
    shared_cfg.simulated_latency_ns = 0;
    shared_cfg.shared = true;
    neuralOS::mesh::CxlMemoryPool shared_pool(shared_cfg);
    CHECK(shared_pool.is_shared(), "shared pool");

    /* Test 9: Domain constants */
    CHECK(neuralOS::mesh::NF_MEM_DOMAIN_CXL == 6, "CXL domain == 6");
    CHECK(neuralOS::mesh::NF_MEM_DOMAIN_CXL_SHARED == 7, "CXL_SHARED domain == 7");

    printf("PASS: all CXL memory tests passed\n");
    return 0;
}
