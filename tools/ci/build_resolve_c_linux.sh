#!/bin/bash
# Build libresolve_c.so inside a manylinux container, from the repository root,
# against the libtorch unpacked at ./libtorch. The image's glibc is the oldest
# the library then loads on; the release workflow runs this in manylinux_2_28
# (glibc 2.28), the baseline of PyTorch's own Linux builds.
#
#   USE_CUDA              ON or OFF (default OFF)
#   CUDA_PATH             the CUDA toolkit, mounted into the container (USE_CUDA=ON)
#   TORCH_CUDA_ARCH_LIST  the CUDA architectures to compile for (USE_CUDA=ON)
#   MAX_GLIBC             the newest glibc symbol version the library may need (default 2.28)
set -euo pipefail

USE_CUDA="${USE_CUDA:-OFF}"
MAX_GLIBC="${MAX_GLIBC:-2.28}"

args=(-S src/core -B build_c
      -DCMAKE_BUILD_TYPE=Release
      -DBUILD_PYTHON=OFF -DBUILD_CLI=OFF -DBUILD_TESTS=OFF
      -DBUILD_R_CAPI=ON -DRESOLVE_USE_OPENMP=OFF
      -DUSE_CUDA="$USE_CUDA"
      -DTorch_DIR="$PWD/libtorch/share/cmake/Torch")
if [ "$USE_CUDA" = ON ]; then
    export PATH="$CUDA_PATH/bin:$PATH"
    args+=(-DTORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" -DCUDAToolkit_ROOT="$CUDA_PATH")
fi

ldd --version | head -1
gcc --version | head -1
cmake "${args[@]}"
cmake --build build_c --target resolve_c --parallel 2

needed=$(objdump -T build_c/libresolve_c.so | grep -o 'GLIBC_[0-9][0-9.]*' | sed 's/GLIBC_//' | sort -uV | tail -1)
echo "libresolve_c.so needs glibc $needed (allowed up to $MAX_GLIBC)"
if [ "$(printf '%s\n%s\n' "$needed" "$MAX_GLIBC" | sort -V | tail -1)" != "$MAX_GLIBC" ]; then
    echo "libresolve_c.so needs glibc $needed, newer than $MAX_GLIBC" >&2
    exit 1
fi
