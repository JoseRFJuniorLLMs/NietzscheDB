#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -f VERSION ]]; then
    echo "❌ VERSION file not found"
    exit 1
fi

VERSION="$(tr -d '[:space:]' < VERSION)"
if [[ -z "$VERSION" ]]; then
    echo "❌ VERSION is empty"
    exit 1
fi

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
ARCHIVE_NAME="nietzsche-db-v$VERSION-$OS-$ARCH.tar.gz"

echo "🚀 Publishing NietzscheDB v$VERSION..."
echo "ℹ️  Host: $OS-$ARCH"

# 1. Release metadata gate
echo "🔎 Checking release metadata..."
python3 scripts/check_release_version.py

# 2. Full portable workspace tests
# Keep this aligned with CI. The generic release path must not require CUDA/cuVS.
echo "🧪 Running workspace tests (portable CPU profile)..."
cargo test --workspace --no-default-features

# 3. Build release binaries
# NietzscheDB 3.2.0's canonical runtime is nietzsche-server. The historical
# nietzsche-baseserver is not the product binary for this release.
echo "🔨 Building release binaries..."
cargo build --release --bin nietzsche-server --no-default-features
cargo build --release -p nietzsche-cli --no-default-features

STAGING_DIR="target/release_pkg"
rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"

cp target/release/nietzsche-server "$STAGING_DIR/"
cp target/release/nietzsche-cli "$STAGING_DIR/"
cp VERSION "$STAGING_DIR/"
cp CHANGELOG.md "$STAGING_DIR/"

# 4. Create archive
echo "📦 Creating release archive: $ARCHIVE_NAME"
tar -czf "$ARCHIVE_NAME" -C "$STAGING_DIR" .
echo "✅ Archive created: $ARCHIVE_NAME"

# 5. Optional Docker publish
#
# Publishing is intentionally opt-in. The old script pushed to historical
# third-party namespaces. Set one or both variables explicitly:
#   DOCKER_IMAGE=your-dockerhub-user/nietzsche-db
#   GHCR_IMAGE=ghcr.io/owner/nietzsche-db
DOCKER_IMAGE="${DOCKER_IMAGE:-}"
GHCR_IMAGE="${GHCR_IMAGE:-}"

TAGS=()
if [[ -n "$DOCKER_IMAGE" ]]; then
    TAGS+=("-t" "$DOCKER_IMAGE:latest" "-t" "$DOCKER_IMAGE:$VERSION")
fi
if [[ -n "$GHCR_IMAGE" ]]; then
    TAGS+=("-t" "$GHCR_IMAGE:latest" "-t" "$GHCR_IMAGE:$VERSION")
fi

if (( ${#TAGS[@]} > 0 )); then
    echo "🐳 Building and pushing Docker image(s)..."
    if ! docker buildx inspect nietzsche-builder >/dev/null 2>&1; then
        docker buildx create --name nietzsche-builder --use
    else
        docker buildx use nietzsche-builder
    fi

    docker buildx build \
        --platform linux/amd64,linux/arm64 \
        "${TAGS[@]}" \
        --push .
    echo "✅ Docker images pushed."
else
    echo "ℹ️  Docker publish skipped. Set DOCKER_IMAGE and/or GHCR_IMAGE to enable it."
fi

# 6. Commit/tag/push
echo "🐙 Preparing Git tag v$VERSION..."
git add VERSION CHANGELOG.md release.sh
git commit -m "chore: release v$VERSION artifacts" || echo "ℹ️  Nothing to commit"

git push origin HEAD

if git rev-parse -q --verify "refs/tags/v$VERSION" >/dev/null 2>&1; then
    echo "ℹ️  Tag v$VERSION already exists. Skipping tag creation."
else
    git tag -a "v$VERSION" -m "NietzscheDB v$VERSION"
fi

git push origin "v$VERSION"
echo "✅ Git tag v$VERSION pushed."
echo "🎉 NietzscheDB v$VERSION release package ready: $ARCHIVE_NAME"
