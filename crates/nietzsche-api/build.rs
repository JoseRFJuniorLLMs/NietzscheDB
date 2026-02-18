fn main() {
    tonic_build::compile_protos("proto/nietzsche.proto")
        .expect("failed to compile proto/nietzsche.proto");
}
