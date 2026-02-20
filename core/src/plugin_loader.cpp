/**
 * @file plugin_loader.cpp
 * @brief Core plugin loader — discovers, loads, and validates plugins.
 *
 * Uses the platform-abstracted dl_open/dl_sym to load .dylib/.so/.dll
 * files, resolve the nf_plugin_register entry point, validate ABI
 * version, and register the provider vtable with the core scheduler.
 */

#include "neurofabric/neuro_fabric_abi.h"
#include "platform/plugin_loader_internal.h"

#include <cstring>
#include <vector>

namespace nf {

struct LoadedPlugin {
    platform::DynLibHandle  lib_handle = nullptr;
    nf_provider             provider   = nullptr;
    nf_provider_vtable      vtable{};
};

static std::vector<LoadedPlugin> s_plugins;

nf_status load_plugin(const char* path) {
    platform::DynLibHandle lib = platform::dl_open(path);
    if (!lib) {
        return NF_ERROR_PLUGIN_LOAD;
    }

    auto entry = reinterpret_cast<nf_plugin_register_fn>(
        platform::dl_sym(lib, NF_PLUGIN_ENTRY_SYMBOL));
    if (!entry) {
        platform::dl_close(lib);
        return NF_ERROR_PLUGIN_LOAD;
    }

    LoadedPlugin lp{};
    lp.lib_handle = lib;

    nf_status st = entry(&lp.vtable, &lp.provider);
    if (st != NF_OK) {
        platform::dl_close(lib);
        return st;
    }

    // ABI version gate
    if (lp.vtable.get_abi_version &&
        lp.vtable.get_abi_version(lp.provider) != NF_ABI_VERSION) {
        if (lp.vtable.shutdown) lp.vtable.shutdown(lp.provider);
        platform::dl_close(lib);
        return NF_ERROR_ABI_MISMATCH;
    }

    s_plugins.push_back(lp);
    return NF_OK;
}

void unload_all_plugins() {
    for (auto& lp : s_plugins) {
        if (lp.vtable.shutdown) {
            lp.vtable.shutdown(lp.provider);
        }
        platform::dl_close(lp.lib_handle);
    }
    s_plugins.clear();
}

} // namespace nf
