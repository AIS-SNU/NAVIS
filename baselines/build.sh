#!/usr/bin/env bash
# Build the baselines into build/baselines/ (all of them by default):
#
#   bash baselines/build.sh [odinann] [freshdiskann] [ipdiskann]
set -euo pipefail
BASELINES=$(cd "$(dirname "$0")" && pwd)
OUT=${BASELINES_BUILD:-$(dirname "$BASELINES")/build/baselines}
JOBS=${JOBS:-$(nproc)}
mkdir -p "$OUT"

# liburing, the same pinned release as the NAVIS build.
LIBURING_VERSION=2.10
LIBURING_SHA256=0a687616a6886cd82b746b79c4e33dc40b8d7c0c6e24d0f6f3fd7cf41886bf53
build_liburing() {
    [[ -f $OUT/liburing/lib/liburing.a ]] && return
    local tarball=$OUT/liburing-$LIBURING_VERSION.tar.gz src=$OUT/liburing-src
    curl -fsSL -o "$tarball" \
        "https://github.com/axboe/liburing/archive/refs/tags/liburing-$LIBURING_VERSION.tar.gz"
    echo "$LIBURING_SHA256  $tarball" | sha256sum -c --quiet -
    rm -rf "$src" && mkdir -p "$src"
    tar -xzf "$tarball" -C "$src" --strip-components=1
    (cd "$src" && ./configure --prefix="$OUT/liburing" >/dev/null &&
        make -C src -j"$JOBS" >/dev/null && make -C src install >/dev/null)
}

# Converters, plus the FreshDiskANN driver once its library exists.
build_tools() {
    local fresh_lib=$OUT/freshdiskann/src/libdiskann_s.a
    [[ -f $fresh_lib ]] || fresh_lib=
    cmake -S "$BASELINES" -B "$OUT/tools" -DCMAKE_BUILD_TYPE=Release \
        -DFRESHDISKANN_LIB="$fresh_lib" >/dev/null
    cmake --build "$OUT/tools" -j"$JOBS"
}

build_odinann() {
    build_liburing
    cmake -S "$BASELINES/odinann/OdinANN" -B "$OUT/odinann" -DLIBURING_DIR="$OUT/liburing" >/dev/null
    cmake --build "$OUT/odinann" -j"$JOBS"
}

build_freshdiskann() {
    cmake -S "$BASELINES/freshdiskann/FreshDiskANN" -B "$OUT/freshdiskann" \
        -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build "$OUT/freshdiskann" -j"$JOBS" --target diskann_s
}

build_ipdiskann() {
    local cargo=${CARGO:-cargo}
    if [[ $($cargo --version 2>/dev/null) != "cargo 1.97.1 "* ]]; then
        echo "error: IP-DiskANN (DiskANN3) needs Rust 1.97.1:" \
             "rustup toolchain install 1.97.1, then CARGO=\"cargo +1.97.1\"" >&2
        exit 1
    fi
    CARGO_TARGET_DIR=$OUT/ipdiskann $cargo build --locked --release \
        --manifest-path "$BASELINES/ipdiskann/Cargo.toml"
}

systems=("$@")
[[ ${#systems[@]} -gt 0 ]] || systems=(odinann freshdiskann ipdiskann)
for s in "${systems[@]}"; do
    case $s in
        odinann|freshdiskann|ipdiskann) "build_$s" ;;
        *) echo "unknown system: $s (odinann, freshdiskann, ipdiskann)" >&2; exit 1 ;;
    esac
done
build_tools
echo "built: ${systems[*]} -> $OUT"
