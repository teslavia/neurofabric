# CMake toolchain file for cross-compiling to RK3588 (aarch64-linux-gnu)
#
# NOTE: Do NOT set CMAKE_SYSROOT here. The cross-compiler ships with its own
# libc/libstdc++ headers. We only add the RKNN sysroot as an extra search path
# via CMAKE_FIND_ROOT_PATH so find_library() can locate librknnrt.so.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_AR           aarch64-linux-gnu-ar)
set(CMAKE_RANLIB       aarch64-linux-gnu-ranlib)

# RKNN sysroot — only for find_library/find_path, not as compiler sysroot
set(CMAKE_FIND_ROOT_PATH $ENV{SYSROOT})

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)
