fn main() {
    // Use the vendored protoc binary so no system installation is required.
    let protoc = protoc_bin_vendored::protoc_bin_path().expect("vendored protoc not found");
    std::env::set_var("PROTOC", protoc);

    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    tonic_build::configure()
        .file_descriptor_set_path(out_dir.join("nietzsche_descriptor.bin"))
        .compile(&["proto/nietzsche.proto"], &["proto/"])
        .expect("failed to compile proto/nietzsche.proto");
}
