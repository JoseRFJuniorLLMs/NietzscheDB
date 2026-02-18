fn main() {
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    tonic_build::configure()
        .file_descriptor_set_path(out_dir.join("nietzsche_descriptor.bin"))
        .compile_protos(&["proto/nietzsche.proto"], &["proto/"])
        .expect("failed to compile proto/nietzsche.proto");
}
