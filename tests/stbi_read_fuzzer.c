// compatibility shim: the oss-fuzz harness moved to local_fuzz/image.c
// (see commit "tests: Use new fuzzer harness for oss-fuzz integration"),
// but the Makefile still builds stbi_read_fuzzer.c. Keep the old name working.
#include "local_fuzz/image.c"
