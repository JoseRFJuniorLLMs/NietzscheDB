#!/usr/bin/env bash
# Generate Python gRPC stubs from nietzsche.proto
#
# Prerequisites:
#   pip install grpcio-tools
#
# Usage:
#   cd sdks/python && bash nietzschedb/generate_protos.sh

set -euo pipefail

PROTO_DIR="../../crates/nietzsche-api/proto"
OUT_DIR="nietzschedb/proto"

python -m grpc_tools.protoc \
    -I"$PROTO_DIR" \
    --python_out="$OUT_DIR" \
    --grpc_python_out="$OUT_DIR" \
    "$PROTO_DIR/nietzsche.proto"

# Fix relative imports for Python 3
sed -i 's/^import nietzsche_pb2/from . import nietzsche_pb2/' "$OUT_DIR/nietzsche_pb2_grpc.py"

echo "Generated Python gRPC stubs in $OUT_DIR/"
