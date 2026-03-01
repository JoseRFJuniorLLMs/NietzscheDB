fn main() {
    // Compile glibc C23 compatibility stubs for systems with glibc < 2.38.
    // Required because ort-sys (ONNX Runtime) references __isoc23_strto*
    // symbols that only exist in glibc >= 2.38.
    #[cfg(target_os = "linux")]
    {
        cc::Build::new()
            .file("glibc_compat.c")
            .compile("glibc_compat");
    }
}
