#!/bin/sh
# Build and run the image noise-injection robustness test.
# Self-contained: uses only the system C compiler and libm. No network,
# no third-party libraries, and no changes to the existing Makefile.
set -e

cd "$(dirname "$0")"

CC="${CC:-cc}"
OUT="${TMPDIR:-/tmp}/test_image_noise.$$"
trap 'rm -f "$OUT" "$OUT.asan"' EXIT INT TERM

echo "== building test_image_noise (optimized build) =="
$CC -std=c99 -O2 -Wall -Wextra -o "$OUT" test_image_noise.c -lm

echo "== running =="
if "$OUT"; then
   rc=0
else
   rc=$?
fi

# Optional extra confidence pass under sanitizers; skipped if the toolchain
# does not support them (never makes the run fail for environmental reasons).
if $CC -std=c99 -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all \
       -o "$OUT.asan" test_image_noise.c -lm >/dev/null 2>&1; then
   echo
   echo "== rerunning under ASan/UBSan =="
   "$OUT.asan" || rc=$?
fi

exit $rc
