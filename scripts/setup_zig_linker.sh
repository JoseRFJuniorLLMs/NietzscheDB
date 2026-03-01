#!/bin/bash
# Setup zig as clang/cc replacement for WSL compilation
# Translates Rust target triples to zig-compatible ones
mkdir -p "$HOME/bin"

cat > "$HOME/bin/clang" << 'EOF'
#!/bin/bash
# Translate Rust target triple to zig-compatible triple
args=()
for arg in "$@"; do
    case "$arg" in
        --target=x86_64-unknown-linux-gnu)
            args+=("--target=x86_64-linux-gnu")
            ;;
        --target=*-unknown-linux-*)
            fixed="${arg//-unknown-linux-/-linux-}"
            args+=("$fixed")
            ;;
        *)
            args+=("$arg")
            ;;
    esac
done
exec "$HOME/zig-linux-x86_64-0.13.0/zig" cc "${args[@]}"
EOF
chmod +x "$HOME/bin/clang"

cp "$HOME/bin/clang" "$HOME/bin/cc"

cat > "$HOME/bin/ar" << 'EOF'
#!/bin/bash
exec "$HOME/zig-linux-x86_64-0.13.0/zig" ar "$@"
EOF
chmod +x "$HOME/bin/ar"

echo "Zig linker wrappers installed (with target translation)"
"$HOME/bin/clang" --version 2>&1 | head -1
echo "Test compile:"
echo 'int main(){return 0;}' > /tmp/test.c
"$HOME/bin/clang" --target=x86_64-linux-gnu -o /tmp/test /tmp/test.c 2>&1 && echo "OK: compile works" || echo "FAIL"
