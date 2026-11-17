/*
 * stbi_read_fuzzer.c -- fuzz harness for stb_image, oss-fuzz style.
 *
 * tests/Makefile links this with fuzz_main.c to build image_fuzzer;
 * fuzz_main.c supplies main() and feeds file contents to
 * LLVMFuzzerTestOneInput(). This snapshot of the repository was missing
 * the harness, which broke the image_fuzzer build target.
 */

#include <stdint.h>
#include <stddef.h>
#include <limits.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#define STBI_FAILURE_USERMSG
#include "stb_image.h"

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
   int x, y, channels;

   if (size == 0 || size > (size_t) INT_MAX)
      return 0;

   stbi_info_from_memory(data, (int) size, &x, &y, &channels);
   stbi_is_16_bit_from_memory(data, (int) size);

   stbi_image_free(stbi_load_from_memory(data, (int) size,
                                         &x, &y, &channels, 0));
   stbi_image_free(stbi_load_16_from_memory(data, (int) size,
                                            &x, &y, &channels, 0));
   stbi_image_free(stbi_loadf_from_memory(data, (int) size,
                                          &x, &y, &channels, 0));
   return 0;
}
