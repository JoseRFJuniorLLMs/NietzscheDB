// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let protoc = protoc_bin_vendored::protoc_bin_path().expect("vendored protoc not found");
    std::env::set_var("PROTOC", protoc);
    tonic_build::compile_protos("proto/nietzsche_db.proto")?;
    Ok(())
}
