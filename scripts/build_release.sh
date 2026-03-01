#!/bin/bash
source "$HOME/.cargo/env" 2>/dev/null || true
export PATH="$HOME/bin:$PATH"
export CC="$HOME/bin/clang"

cd /mnt/d/DEV/NietzscheDB

# Clean stale build artifacts from failed previous builds
cargo clean -p bzip2-sys -p lz4-sys -p libz-sys -p librocksdb-sys 2>/dev/null || true

echo "Clang: $(which clang)"
echo "Cargo: $(which cargo)"
echo "Rustc: $(rustc --version)"
echo "Starting release build..."

cargo build --release 2>&1

RC=$?
echo "---BUILD_DONE--- (exit: $RC)"

if [ -f target/release/nietzsche-server ]; then
    echo "SUCCESS!"
    ls -lh target/release/nietzsche-server
    file target/release/nietzsche-server
else
    echo "Binary not found"
    ls target/release/nietzsche* 2>/dev/null | head -5
fi
