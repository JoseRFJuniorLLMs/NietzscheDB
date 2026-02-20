# ── Stage 1: builder ─────────────────────────────────────────────────────────
#
# Build nietzsche-server with full LTO + stripped symbols.
# RocksDB links statically via the `rocksdb` crate — no shared .so needed at
# runtime.
FROM rustlang/rust:nightly-slim AS builder

# System dependencies for RocksDB + tonic-build (protoc)
RUN apt-get update && apt-get install -y --no-install-recommends \
        protobuf-compiler \
        libclang-dev \
        clang \
        cmake \
        pkg-config \
        libssl-dev \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy workspace manifests (Cargo.lock* makes it optional if not committed).
COPY Cargo.toml Cargo.lock* ./
COPY crates/ crates/

# Build the production binary.
# [profile.release] is already configured in workspace Cargo.toml:
#   lto = true, codegen-units = 1, strip = true, panic = "abort"
RUN cargo build --release --bin nietzsche-server

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM ubuntu:24.04 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libssl3 \
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -ms /bin/bash nietzsche
USER nietzsche
WORKDIR /home/nietzsche

# Stripped production binary
COPY --from=builder /build/target/release/nietzsche-server ./nietzsche-server

# ── Configuration (override at docker run / Kubernetes env) ──────────────────
ENV NIETZSCHE_DATA_DIR=/data/nietzsche
ENV NIETZSCHE_PORT=50052
ENV NIETZSCHE_LOG_LEVEL=info
ENV NIETZSCHE_SLEEP_INTERVAL_SECS=300
ENV NIETZSCHE_SLEEP_NOISE=0.02
ENV NIETZSCHE_SLEEP_ADAM_STEPS=10
ENV NIETZSCHE_HAUSDORFF_THRESHOLD=0.15
ENV NIETZSCHE_MAX_CONNECTIONS=1024
ENV NIETZSCHE_DASHBOARD_PORT=8080

# Persistent data directory — mount a volume here
VOLUME ["/data/nietzsche"]

# gRPC port + HTTP dashboard
EXPOSE 50052 8080

LABEL org.opencontainers.image.title="NietzscheDB Server"
LABEL org.opencontainers.image.description="Temporal Hyperbolic Graph Database — production gRPC server"
LABEL org.opencontainers.image.source="https://github.com/JoseRFJuniorLLMs/NietzscheDB"

ENTRYPOINT ["./nietzsche-server"]
