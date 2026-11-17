#!/bin/sh
# Build and run the stbi_view flipped-view tests: normal build, then an
# AddressSanitizer build (which also runs the dangling-view child check).
# Standalone: does not touch the repo Makefile.
set -e
cd "$(dirname "$0")"

OUT="${TMPDIR:-/tmp}/stb_view_flip_test"
mkdir -p "$OUT"

echo "--- normal build (-O2) ---"
${CC:-cc} -I.. -O2 -Wall -Wextra -Wno-pointer-to-int-cast -Wno-int-to-pointer-cast \
   view_flip_test.c -lm -o "$OUT/view_flip_test"
"$OUT/view_flip_test"

echo "--- AddressSanitizer build (-O1 -g) ---"
${CC:-cc} -I.. -O1 -g -fsanitize=address -Wall -Wextra \
   -Wno-pointer-to-int-cast -Wno-int-to-pointer-cast \
   view_flip_test.c -lm -o "$OUT/view_flip_test_asan"
"$OUT/view_flip_test_asan"
