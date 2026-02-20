/**
 * @file plugin_loader_internal.h
 * @brief Internal abstraction for OS-specific dynamic library loading.
 *
 * NOT part of the public ABI. Used only within core/src/.
 */

#ifndef NF_PLUGIN_LOADER_INTERNAL_H
#define NF_PLUGIN_LOADER_INTERNAL_H

namespace nf {
namespace platform {

/** Opaque handle to a loaded dynamic library. */
using DynLibHandle = void*;

/** Load a shared library from `path`. Returns nullptr on failure. */
DynLibHandle dl_open(const char* path);

/** Resolve `symbol` in the given library. Returns nullptr if not found. */
void* dl_sym(DynLibHandle handle, const char* symbol);

/** Unload a library. NULL-safe. */
void dl_close(DynLibHandle handle);

/** Last error message from the platform loader. May return nullptr. */
const char* dl_error();

} // namespace platform
} // namespace nf

#endif // NF_PLUGIN_LOADER_INTERNAL_H
