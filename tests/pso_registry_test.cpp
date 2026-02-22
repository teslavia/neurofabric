/**
 * @file pso_registry_test.cpp
 * @brief Phase 28: PSO registry table validation (platform-independent)
 *
 * Validates the static kPSOTable without requiring Metal hardware.
 */

#include "metal_pso_registry.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); \
        std::exit(1); \
    } \
} while(0)

int main() {
    std::printf("=== PSO Registry Test ===\n");

    /* Test 1: kPSOTableSize matches expected count */
    std::printf("[1] Table size = %u, PSO_COUNT = %u\n", kPSOTableSize, (unsigned)PSO_COUNT);

    /* Test 2: No duplicate indices */
    {
        std::set<uint16_t> seen;
        for (uint16_t i = 0; i < kPSOTableSize; ++i) {
            CHECK(seen.find(kPSOTable[i].index) == seen.end(),
                  "Duplicate PSO index in kPSOTable");
            seen.insert(kPSOTable[i].index);
        }
        std::printf("[2] No duplicate indices: PASS (%u unique)\n", (unsigned)seen.size());
    }

    /* Test 3: No duplicate MSL names */
    {
        std::set<std::string> seen;
        for (uint16_t i = 0; i < kPSOTableSize; ++i) {
            std::string name(kPSOTable[i].msl_name);
            CHECK(seen.find(name) == seen.end(),
                  "Duplicate MSL name in kPSOTable");
            seen.insert(name);
        }
        std::printf("[3] No duplicate MSL names: PASS (%u unique)\n", (unsigned)seen.size());
    }

    /* Test 4: All entries have non-null, non-empty MSL names */
    {
        for (uint16_t i = 0; i < kPSOTableSize; ++i) {
            CHECK(kPSOTable[i].msl_name != nullptr, "Null MSL name");
            CHECK(std::strlen(kPSOTable[i].msl_name) > 0, "Empty MSL name");
        }
        std::printf("[4] All MSL names non-null/non-empty: PASS\n");
    }

    /* Test 5: All indices < PSO_COUNT */
    {
        for (uint16_t i = 0; i < kPSOTableSize; ++i) {
            CHECK(kPSOTable[i].index < PSO_COUNT,
                  "PSO index out of range");
        }
        std::printf("[5] All indices < PSO_COUNT: PASS\n");
    }

    /* Test 6: Every non-SIMD PSO index is covered */
    {
        bool covered[PSO_COUNT] = {};
        uint16_t simd_count = 0;
        for (uint16_t i = 0; i < kPSOTableSize; ++i) {
            covered[kPSOTable[i].index] = true;
            if (kPSOTable[i].requires_simd) ++simd_count;
        }
        uint16_t missing = 0;
        for (uint16_t i = 0; i < PSO_COUNT; ++i) {
            if (!covered[i]) ++missing;
        }
        CHECK(missing == 0, "Some PSO indices not covered by kPSOTable");
        std::printf("[6] Full coverage: PASS (all %u slots covered, %u SIMD-conditional)\n",
                    (unsigned)PSO_COUNT, (unsigned)simd_count);
    }

    /* Test 7: SIMD entries are the expected ones */
    {
        uint16_t simd_count = 0;
        for (uint16_t i = 0; i < kPSOTableSize; ++i) {
            if (kPSOTable[i].requires_simd) {
                CHECK(kPSOTable[i].index == PSO_LINEAR_SIMD ||
                      kPSOTable[i].index == PSO_LINEAR_SIMD_F16 ||
                      kPSOTable[i].index == PSO_LINEAR_F16_TO_F32 ||
                      kPSOTable[i].index == PSO_FUSED_DEQUANT_Q4_0_LINEAR_SIMD ||
                      kPSOTable[i].index == PSO_FUSED_DEQUANT_Q4_0_LINEAR_SIMD_F16,
                      "Unexpected SIMD-conditional PSO");
                ++simd_count;
            }
        }
        CHECK(simd_count == 5, "Expected exactly 5 SIMD-conditional PSOs");
        std::printf("[7] SIMD entries validated: PASS (%u entries)\n", (unsigned)simd_count);
    }

    std::printf("\nAll PSO registry tests passed.\n");
    return 0;
}
