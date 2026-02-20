/**
 * @file smoke_test.cpp
 * @brief Minimal compile + link smoke test for nf_core.
 */

#include "neurofabric/neuro_fabric_abi.h"
#include <cstdio>
#include <cstring>

int main() {
    // Verify ABI version packing
    static_assert(NF_ABI_VERSION == 0x000100,
                  "ABI version packing broken");

    // Verify tensor descriptor is POD and memcpy-safe
    nf_tensor_desc desc{};
    desc.dtype      = NF_DTYPE_F32;
    desc.ndim       = 2;
    desc.shape[0]   = 4;
    desc.shape[1]   = 8;
    desc.size_bytes = 4 * 8 * sizeof(float);

    nf_tensor_desc copy;
    std::memcpy(&copy, &desc, sizeof(nf_tensor_desc));

    if (copy.ndim != 2 || copy.shape[1] != 8) {
        std::fprintf(stderr, "FAIL: tensor_desc memcpy broken\n");
        return 1;
    }

    // Verify vtable struct is zero-initializable
    nf_provider_vtable vt{};
    if (vt.get_name != nullptr) {
        std::fprintf(stderr, "FAIL: vtable zero-init broken\n");
        return 1;
    }

    std::printf("OK: neuro_fabric_abi smoke test passed\n");
    return 0;
}
