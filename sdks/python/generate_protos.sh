#!/bin/bash
set -e

# Run from sdks/python/
cd "$(dirname "$0")"

# Activate venv if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

python3 -m grpc_tools.protoc \
    -I ../../crates/nietzsche-proto/proto \
    --python_out=nietzsche/proto \
    --grpc_python_out=nietzsche/proto \
    ../../crates/nietzsche-proto/proto/nietzsche_db.proto

# Fix import in generated grpc file
if [[ "$OSTYPE" == "darwin"* ]]; then
  sed -i '' 's/import nietzsche_db_pb2 as nietzsche_db__pb2/from . import nietzsche_db_pb2 as nietzsche_db__pb2/' nietzsche/proto/nietzsche_db_pb2_grpc.py
else
  sed -i 's/import nietzsche_db_pb2 as nietzsche_db__pb2/from . import nietzsche_db_pb2 as nietzsche_db__pb2/' nietzsche/proto/nietzsche_db_pb2_grpc.py
fi

echo "Python protos generated and patched."
