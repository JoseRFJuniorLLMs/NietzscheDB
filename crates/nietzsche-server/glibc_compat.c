// glibc C23 compatibility stubs for Ubuntu 22.04 (glibc 2.35).
// ort-sys (ONNX Runtime) emits calls to __isoc23_strto* which only
// exist in glibc >= 2.38.  These thin wrappers forward to the
// standard C11 equivalents that are always available.

#include <stdlib.h>

long __isoc23_strtol(const char *nptr, char **endptr, int base) {
    return strtol(nptr, endptr, base);
}

long long __isoc23_strtoll(const char *nptr, char **endptr, int base) {
    return strtoll(nptr, endptr, base);
}

unsigned long __isoc23_strtoul(const char *nptr, char **endptr, int base) {
    return strtoul(nptr, endptr, base);
}

unsigned long long __isoc23_strtoull(const char *nptr, char **endptr, int base) {
    return strtoull(nptr, endptr, base);
}
