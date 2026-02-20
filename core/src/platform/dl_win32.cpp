/**
 * @file dl_win32.cpp
 * @brief Win32 dynamic library loader
 *
 * Wraps LoadLibrary/GetProcAddress/FreeLibrary behind the internal
 * DynLib interface. Compiled ONLY on Windows (see core/CMakeLists.txt).
 */

#include "plugin_loader_internal.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace nf {
namespace platform {

DynLibHandle dl_open(const char* path) {
    HMODULE h = ::LoadLibraryA(path);
    return static_cast<DynLibHandle>(h);
}

void* dl_sym(DynLibHandle handle, const char* symbol) {
    FARPROC p = ::GetProcAddress(static_cast<HMODULE>(handle), symbol);
    // FARPROC -> void* is safe on Windows; both are pointer-sized.
    return reinterpret_cast<void*>(p);
}

void dl_close(DynLibHandle handle) {
    if (handle) ::FreeLibrary(static_cast<HMODULE>(handle));
}

const char* dl_error() {
    // Simplified: return nullptr; real impl would FormatMessage.
    return nullptr;
}

} // namespace platform
} // namespace nf
