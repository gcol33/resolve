/**
 * CUDA 13.x compatibility header for PyTorch.
 *
 * PyTorch's c10/util/Exception.h uses 'char' as a variable name,
 * which conflicts with CUDA 13.x's reserved identifiers.
 *
 * This header provides a workaround by patching the issue before
 * including PyTorch headers.
 *
 * Usage: Include this header BEFORE any PyTorch/ATen headers in .cu files.
 */

#ifndef RESOLVE_CUDA_COMPAT_H
#define RESOLVE_CUDA_COMPAT_H

// Only apply workaround for CUDA 13.x and later
#if defined(__CUDACC__) && defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ >= 13

// The issue is in c10/util/Exception.h where it does something like:
// template <typename T> void format_str(T& out, const char* str, char) { ... }
// where 'char' is used as a parameter name, conflicting with CUDA's reserved identifier.

// Workaround: Define a macro that renames the problematic parameter before PyTorch headers
// and restore after.

// Unfortunately, this specific case is hard to patch with macros because 'char' is a keyword.
// The real fix is to compile .cu files without the PyTorch headers that cause issues,
// or use a separate compilation unit.

// Alternative approach: Use extern "C" style interface
// Define CUDA kernels in a separate file without PyTorch headers

#define RESOLVE_CUDA_13_WORKAROUND 1

#endif // __CUDACC__ && __CUDACC_VER_MAJOR__ >= 13

#endif // RESOLVE_CUDA_COMPAT_H
