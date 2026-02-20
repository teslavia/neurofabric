/**
 * @file dl_posix.cpp
 * @brief POSIX dynamic library loader (macOS / Linux)
 *
 * Wraps dlopen/dlsym/dlclose behind the internal DynLib interface.
 * This file is compiled ONLY on non-Windows targets (see core/CMakeLists.txt).
 */

#include "plugin_loader_internal.h"
#include <dlfcn.h>

namespace nf::platform {

DynLibHandle dl_open(const char* path) {
    void* h = ::dlopen(path, RTLD_NOW | RTLD_LOCAL);
    return static_cast<DynLibHandle>(h);
}

void* dl_sym(DynLibHandle handle, const char* symbol) {
    return ::dlsym(handle, symbol);
}

void dl_close(DynLibHandle handle) {
    if (handle) ::dlclose(handle);
}

const char* dl_error() {
    return ::dlerror();
}

} // namespace nf::platform
