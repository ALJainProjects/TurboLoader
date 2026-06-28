#!/usr/bin/env bash
# Build TurboLoader's C dependencies (libjpeg-turbo, libpng, libwebp, lz4) FROM SOURCE
# at a FIXED macOS deployment target, so the resulting wheels are portable.
#
# Homebrew bottles are compiled for the runner's current macOS, so wheels built against
# them only install on that macOS or newer (and delocate rejects an older target). By
# building the deps ourselves at MACOSX_DEPLOYMENT_TARGET=11.0 the bundled dylibs carry
# an 11.0 minimum and the wheel works on macOS 11+.
#
# Env:
#   TURBOLOADER_DEPS_PREFIX   install prefix (default /usr/local)      — read by setup.py
#   MACOSX_DEPLOYMENT_TARGET  deployment target (default 11.0)
#   CIBW_ARCH / ARCH          arm64 | x86_64 (default: uname -m)
set -euo pipefail

PREFIX="${TURBOLOADER_DEPS_PREFIX:-/usr/local}"
TARGET="${MACOSX_DEPLOYMENT_TARGET:-11.0}"
ARCH="${CIBW_ARCH:-${ARCH:-$(uname -m)}}"
JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
WORK="$(mktemp -d)"

JPEG_TURBO_VER="3.0.4"
LIBPNG_VER="1.6.44"
LIBWEBP_VER="1.4.0"
LZ4_VER="1.10.0"

echo "==> Building macOS deps from source"
echo "    prefix=$PREFIX  target=$TARGET  arch=$ARCH  jobs=$JOBS"
mkdir -p "$PREFIX"

CMAKE_COMMON=(
  -DCMAKE_INSTALL_PREFIX="$PREFIX"
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$TARGET"
  -DCMAKE_OSX_ARCHITECTURES="$ARCH"
  -DBUILD_SHARED_LIBS=ON
)

fetch() {  # url  outfile
  echo "==> Fetching $1"
  curl -fsSL --retry 3 -o "$WORK/$2" "$1"
}

cd "$WORK"

# ---- libjpeg-turbo (cmake) ----
fetch "https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/${JPEG_TURBO_VER}/libjpeg-turbo-${JPEG_TURBO_VER}.tar.gz" jt.tgz
tar xf jt.tgz
cmake -S "libjpeg-turbo-${JPEG_TURBO_VER}" -B build-jt "${CMAKE_COMMON[@]}" \
  -DENABLE_SHARED=ON -DENABLE_STATIC=OFF -DWITH_TURBOJPEG=OFF
cmake --build build-jt --target install -j "$JOBS"

# ---- lz4 (cmake project lives under build/cmake) ----
fetch "https://github.com/lz4/lz4/releases/download/v${LZ4_VER}/lz4-${LZ4_VER}.tar.gz" lz4.tgz
tar xf lz4.tgz
cmake -S "lz4-${LZ4_VER}/build/cmake" -B build-lz4 "${CMAKE_COMMON[@]}"
cmake --build build-lz4 --target install -j "$JOBS"

# ---- libpng (cmake; uses the system zlib) ----
fetch "https://download.sourceforge.net/libpng/libpng-${LIBPNG_VER}.tar.gz" png.tgz
tar xf png.tgz
cmake -S "libpng-${LIBPNG_VER}" -B build-png "${CMAKE_COMMON[@]}" \
  -DPNG_SHARED=ON -DPNG_STATIC=OFF -DPNG_TESTS=OFF -DPNG_EXECUTABLES=OFF
cmake --build build-png --target install -j "$JOBS"

# ---- libwebp (cmake; builds libwebp + libsharpyuv) ----
fetch "https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-${LIBWEBP_VER}.tar.gz" webp.tgz
tar xf webp.tgz
cmake -S "libwebp-${LIBWEBP_VER}" -B build-webp "${CMAKE_COMMON[@]}" \
  -DWEBP_BUILD_ANIM_UTILS=OFF -DWEBP_BUILD_CWEBP=OFF -DWEBP_BUILD_DWEBP=OFF \
  -DWEBP_BUILD_GIF2WEBP=OFF -DWEBP_BUILD_IMG2WEBP=OFF -DWEBP_BUILD_VWEBP=OFF \
  -DWEBP_BUILD_WEBPINFO=OFF -DWEBP_BUILD_WEBPMUX=OFF -DWEBP_BUILD_EXTRAS=OFF
cmake --build build-webp --target install -j "$JOBS"

echo "==> Done. Installed into $PREFIX/lib:"
ls -1 "$PREFIX"/lib/lib{jpeg,png,webp,lz4,sharpyuv}*.dylib 2>/dev/null || true
rm -rf "$WORK"
