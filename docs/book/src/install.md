# Installation

NietzscheDB runs on Linux and macOS. Windows is supported via WSL2.

## Prerequisites
*   **Rust**: Nightly toolchain is required for SIMD features.
*   **Protoc**: Protocol Buffer compiler for gRPC.

## Option 1: Docker (Recommended)

The easiest way to get started.

```bash
docker pull glukhota/nietzsche-db:latest
# or build locally
docker build -t nietzschedb .

docker run -p 50051:50051 -v $(pwd)/data:/app/data nietzschedb
```

## Option 2: Build from Source

1.  **Install dependencies**
    ```bash
    # Ubuntu/Debian
    sudo apt install protobuf-compiler cmake

    # macOS
    brew install protobuf
    ```

2.  **Install Rust Nightly**
    ```bash
    rustup toolchain install nightly
    rustup default nightly
    ```

3.  **Clone and Build**
    ```bash
    git clone https://github.com/yarlabs/nietzsche-db
    cd nietzsche-db
    cargo build --release
    ```

4.  **Run**
    ```bash
    ./target/release/nietzsche-baseserver
    ```
