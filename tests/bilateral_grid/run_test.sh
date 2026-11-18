#!/bin/sh
# Build and run the bilateral-grid acceptance test.
# Uses only the C compiler and stb headers already present in the repository.
set -e
cd "$(dirname "$0")"
mkdir -p artifacts_bilateral
cc -std=c11 -O2 -Wall -Wextra -Wpedantic -Wshadow \
   test_bilateral_grid.c -lm -o test_bilateral_grid
./test_bilateral_grid
