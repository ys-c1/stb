// test_view_crop.c - tests for the stbi_view sub-image view API in stb_image.h
//
// Covers, for a synthetic 95x64 RGB image with a 16-byte-aligned padded stride:
//   - four rectangle groups: interior, boundary-crossing, fully-outside, zero-area
//   - both out-of-bounds modes (clamp / fill) with configurable fill color
//   - configurable stride alignment (accept + reject cases)
//   - pixel-exact consistency against an independent copy-crop reference
//   - zero allocations during view creation and reads (counting allocator)
//   - view shares memory with the source buffer (write-through observation)
//   - timing: view creation/traversal vs copy-crop
//   - decode unchanged: PNG roundtrip through stbi_load is pixel-exact
//
// build (from repo root):  cc -I. -O2 -Wall -Wextra tests/test_view_crop.c -lm -o tests/test_view_crop

#define _POSIX_C_SOURCE 199309L // clock_gettime

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ---- counting allocator: proves the view API performs zero allocations ----
static long g_allocs = 0;
static long g_alloc_bytes = 0;
static void *cnt_malloc(size_t n) { ++g_allocs; g_alloc_bytes += (long) n; return malloc(n); }
static void *cnt_realloc(void *p, size_t n) { if (!p) { ++g_allocs; g_alloc_bytes += (long) n; } return realloc(p, n); }
static void cnt_free(void *p) { free(p); }

#define STBI_MALLOC(sz)     cnt_malloc(sz)
#define STBI_REALLOC(p,sz)  cnt_realloc(p,sz)
#define STBI_FREE(p)        cnt_free(p)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define IMG_W 95
#define IMG_H 64
#define IMG_C 3
#define IMG_STRIDE 288              // 95*3 = 285 padded up to a multiple of 16
#define ALIGN 16

static stbi_uc g_img[IMG_STRIDE * IMG_H];

static int g_failures = 0;

#define CHECK(cond, msg) do { \
   if (!(cond)) { ++g_failures; printf("  FAIL: %s\n", msg); } \
} while (0)

static double now_sec(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
}

static void fill_pattern(void)
{
   int x, y;
   memset(g_img, 0, sizeof(g_img));
   for (y = 0; y < IMG_H; ++y)
      for (x = 0; x < IMG_W; ++x) {
         stbi_uc *p = g_img + y * IMG_STRIDE + x * IMG_C;
         p[0] = (stbi_uc) (x * 3 + y);
         p[1] = (stbi_uc) (y * 5 + (x ^ (y * 7)));
         p[2] = (stbi_uc) ((x + y * IMG_W) & 0xff);
      }
}

// independent copy-crop reference: allocates a packed dst buffer and copies
// with the same clamp/fill semantics; returns NULL on the same error cases
static stbi_uc *copy_crop(const stbi_uc *src, int img_w, int img_h, int comp, int stride,
                          int x, int y, int w, int h, int oob_mode, const stbi_uc *fill,
                          int *out_w, int *out_h)
{
   int ix0 = x < 0 ? 0 : x, iy0 = y < 0 ? 0 : y;
   int ix1 = x + w > img_w ? img_w : x + w, iy1 = y + h > img_h ? img_h : y + h;
   int cw, ch, px, py, c;
   int ox, oy;
   stbi_uc *dst;
   if (w <= 0 || h <= 0) return NULL;
   if (ix0 >= ix1 || iy0 >= iy1) return NULL;
   cw = (oob_mode == STBI_VIEW_OOB_CLAMP) ? ix1 - ix0 : w;
   ch = (oob_mode == STBI_VIEW_OOB_CLAMP) ? iy1 - iy0 : h;
   ox = (oob_mode == STBI_VIEW_OOB_CLAMP) ? ix0 : x; // clamp mode reads from the clamped origin
   oy = (oob_mode == STBI_VIEW_OOB_CLAMP) ? iy0 : y;
   dst = (stbi_uc *) malloc((size_t) cw * ch * comp);
   for (py = 0; py < ch; ++py)
      for (px = 0; px < cw; ++px) {
         int sx = ox + px, sy = oy + py;
         stbi_uc *d = dst + ((size_t) py * cw + px) * comp;
         if (sx >= 0 && sx < img_w && sy >= 0 && sy < img_h)
            memcpy(d, src + (size_t) sy * stride + (size_t) sx * comp, (size_t) comp);
         else
            for (c = 0; c < comp; ++c) d[c] = fill[c];
      }
   *out_w = cw;
   *out_h = ch;
   return dst;
}

// compare full view (via stbi_view_read) against the copy-crop reference
static int view_matches_ref(const stbi_view *v, const stbi_uc *ref)
{
   int px, py, mismatches = 0;
   for (py = 0; py < v->height; ++py)
      for (px = 0; px < v->width; ++px) {
         stbi_uc got[4];
         stbi_view_read(v, px, py, got);
         if (memcmp(got, ref + ((size_t) py * v->width + px) * IMG_C, IMG_C) != 0)
            ++mismatches;
      }
   return mismatches;
}

static unsigned long view_checksum_direct(const stbi_view *v)
{
   // traverse the in-bounds content block straight through the shared pointer
   unsigned long sum = 0;
   int px, py;
   for (py = 0; py < v->content_h; ++py) {
      const stbi_uc *row = v->data + (size_t) py * v->stride;
      for (px = 0; px < v->content_w * v->channels; ++px)
         sum += row[px];
   }
   return sum;
}

struct rect_case {
   const char *name;
   int x, y, w, h;
};

static void run_rect_group(const char *group, const struct rect_case *cases, int n,
                           const stbi_uc *fill, int expect_ok)
{
   int ci, mode;
   printf("== %s ==\n", group);
   for (ci = 0; ci < n; ++ci) {
      const struct rect_case *rc = &cases[ci];
      for (mode = STBI_VIEW_OOB_CLAMP; mode <= STBI_VIEW_OOB_FILL; ++mode) {
         stbi_view_options opt;
         stbi_view v;
         int ok;
         opt.alignment = ALIGN;
         opt.oob_mode = mode;
         memcpy(opt.fill, fill, 4);

         ok = stbi_view_init(&v, g_img, IMG_W, IMG_H, IMG_C, IMG_STRIDE,
                             rc->x, rc->y, rc->w, rc->h, &opt);
         printf("  rect (%d,%d %dx%d) mode=%s -> %s",
                rc->x, rc->y, rc->w, rc->h,
                mode == STBI_VIEW_OOB_CLAMP ? "clamp" : "fill ",
                ok ? "ok" : "ERROR");
         if (!expect_ok) {
            CHECK(!ok, "expected error but view was created");
            printf("\n");
            continue;
         }
         CHECK(ok, "expected success but got error");
         if (!ok) { printf("\n"); continue; }

         printf("  view=%dx%d origin=(%d,%d) stride=%d content=(%d,%d %dx%d)\n",
                v.width, v.height, v.origin_x, v.origin_y, v.stride,
                v.content_x, v.content_y, v.content_w, v.content_h);

         // view must point into the source buffer: zero-copy, zero-alloc
         CHECK(v.data >= g_img && v.data < g_img + sizeof(g_img),
               "view data pointer is not inside the source buffer");
         CHECK(v.stride == IMG_STRIDE, "view stride differs from source stride");

         // consistency with the copy-crop reference
         {
            int rw = 0, rh = 0, mm;
            stbi_uc *ref = copy_crop(g_img, IMG_W, IMG_H, IMG_C, IMG_STRIDE,
                                     rc->x, rc->y, rc->w, rc->h, mode, fill, &rw, &rh);
            CHECK(ref != NULL, "reference copy-crop failed unexpectedly");
            CHECK(rw == v.width && rh == v.height, "view size differs from copy-crop size");
            mm = ref ? view_matches_ref(&v, ref) : -1;
            CHECK(mm == 0, "view pixels differ from copy-crop reference");
            printf("    copy-crop consistency: %s (%d mismatches, %d px compared)\n",
                   mm == 0 ? "MATCH" : "DIFF", mm, v.width * v.height);
            free(ref);
         }

         // shared memory: write through the source, observe through the view
         {
            int cx = v.content_x, cy = v.content_y;
            stbi_uc got[4], save[IMG_C];
            int sx = v.origin_x + cx, sy = v.origin_y + cy;
            stbi_uc *src_px = g_img + (size_t) sy * IMG_STRIDE + (size_t) sx * IMG_C;
            memcpy(save, src_px, IMG_C);
            src_px[0] = 0xAB; src_px[1] = 0xCD; src_px[2] = 0xEF;
            stbi_view_read(&v, cx, cy, got);
            CHECK(got[0] == 0xAB && got[1] == 0xCD && got[2] == 0xEF,
                  "view does not share memory with source buffer");
            memcpy(src_px, save, IMG_C); // restore
         }

         // out-of-bounds behavior
         if (mode == STBI_VIEW_OOB_FILL) {
            // read a pixel inside the view but outside the image (if any)
            int found = 0, px, py;
            for (py = 0; py < v.height && !found; ++py)
               for (px = 0; px < v.width && !found; ++px) {
                  int sx = v.origin_x + px, sy = v.origin_y + py;
                  if (sx < 0 || sx >= IMG_W || sy < 0 || sy >= IMG_H) {
                     stbi_uc got[4];
                     stbi_view_read(&v, px, py, got);
                     CHECK(got[0] == fill[0] && got[1] == fill[1] && got[2] == fill[2],
                           "OOB pixel does not return fill color");
                     printf("    OOB pixel at view (%d,%d) [src (%d,%d)] -> fill {%d,%d,%d}\n",
                            px, py, sx, sy, got[0], got[1], got[2]);
                     found = 1;
                  }
               }
            if (!found)
               printf("    (rect fully inside image: no OOB pixels in fill mode)\n");
         }
         // reading outside the view itself must also yield fill, never crash
         {
            stbi_uc got[4];
            stbi_view_read(&v, -1, -1, got);
            CHECK(got[0] == fill[0] && got[1] == fill[1] && got[2] == fill[2],
                  "read outside view bounds does not return fill color");
            stbi_view_read(&v, v.width, v.height, got);
            CHECK(got[0] == fill[0], "read past view corner does not return fill color");
         }
      }
   }
}

static void run_alignment_cases(void)
{
   stbi_view_options opt;
   stbi_view v;
   static const stbi_uc fill[4] = { 0, 0, 0, 0 };
   int ok;

   printf("== stride alignment (configurable) ==\n");
   memset(&opt, 0, sizeof(opt));
   memcpy(opt.fill, fill, 4);

   opt.alignment = 16;
   ok = stbi_view_init(&v, g_img, IMG_W, IMG_H, IMG_C, IMG_STRIDE, 0, 0, 10, 10, &opt);
   printf("  stride=288 align=16 -> %s (expect ok)\n", ok ? "ok" : "ERROR");
   CHECK(ok, "aligned stride rejected");

   opt.alignment = 32;
   ok = stbi_view_init(&v, g_img, IMG_W, IMG_H, IMG_C, IMG_STRIDE, 0, 0, 10, 10, &opt);
   printf("  stride=288 align=32 -> %s (expect ok)\n", ok ? "ok" : "ERROR");
   CHECK(ok, "aligned stride rejected");

   opt.alignment = 64;
   ok = stbi_view_init(&v, g_img, IMG_W, IMG_H, IMG_C, IMG_STRIDE, 0, 0, 10, 10, &opt);
   printf("  stride=288 align=64 -> %s (expect ERROR: 288 %% 64 != 0)\n", ok ? "ok" : "ERROR");
   CHECK(!ok, "misaligned stride accepted");

   opt.alignment = 16;
   ok = stbi_view_init(&v, g_img, IMG_W, IMG_H, IMG_C, IMG_W * IMG_C, 0, 0, 10, 10, &opt);
   printf("  stride=285 align=16 -> %s (expect ERROR: packed stride not aligned)\n", ok ? "ok" : "ERROR");
   CHECK(!ok, "unaligned packed stride accepted");

   opt.alignment = 0; // no requirement
   ok = stbi_view_init(&v, g_img, IMG_W, IMG_H, IMG_C, IMG_W * IMG_C, 0, 0, 10, 10, &opt);
   printf("  stride=285 align=0  -> %s (expect ok: no requirement)\n", ok ? "ok" : "ERROR");
   CHECK(ok, "packed stride with no alignment requirement rejected");

   ok = stbi_view_init(&v, g_img, IMG_W, IMG_H, IMG_C, IMG_W * IMG_C - 1, 0, 0, 10, 10, NULL);
   printf("  stride=284 (< w*c)  -> %s (expect ERROR: stride too small)\n", ok ? "ok" : "ERROR");
   CHECK(!ok, "too-small stride accepted");
}

static void run_timing(const stbi_uc *fill)
{
   enum { CREATE_ITERS = 1000000, TRAV_ITERS = 20000 };
   stbi_view_options opt;
   stbi_view v;
   double t0, t1;
   unsigned long sink = 0;
   int i, px, py;
   struct rect_case { int x, y, w, h; } rc = { 10, 8, 40, 30 };

   opt.alignment = ALIGN;
   opt.oob_mode = STBI_VIEW_OOB_CLAMP;
   memcpy(opt.fill, fill, 4);

   printf("== timing (rect %dx%d at (%d,%d), clamp mode) ==\n", rc.w, rc.h, rc.x, rc.y);

   t0 = now_sec();
   for (i = 0; i < CREATE_ITERS; ++i) {
      stbi_view_init(&v, g_img, IMG_W, IMG_H, IMG_C, IMG_STRIDE, rc.x, rc.y, rc.w, rc.h, &opt);
      sink += (unsigned long) v.width;
   }
   t1 = now_sec();
   printf("  view create only:        %8.1f ns/op  (%d iters)\n",
          (t1 - t0) * 1e9 / CREATE_ITERS, CREATE_ITERS);

   t0 = now_sec();
   for (i = 0; i < TRAV_ITERS; ++i) {
      stbi_view_init(&v, g_img, IMG_W, IMG_H, IMG_C, IMG_STRIDE, rc.x, rc.y, rc.w, rc.h, &opt);
      sink += view_checksum_direct(&v);
   }
   t1 = now_sec();
   printf("  view create+traverse:    %8.1f ns/op  (%d iters, direct pointer)\n",
          (t1 - t0) * 1e9 / TRAV_ITERS, TRAV_ITERS);

   t0 = now_sec();
   for (i = 0; i < TRAV_ITERS; ++i) {
      stbi_view_init(&v, g_img, IMG_W, IMG_H, IMG_C, IMG_STRIDE, rc.x, rc.y, rc.w, rc.h, &opt);
      for (py = 0; py < v.height; ++py)
         for (px = 0; px < v.width; ++px) {
            stbi_uc got[4];
            stbi_view_read(&v, px, py, got);
            sink += got[0] + got[1] + got[2];
         }
   }
   t1 = now_sec();
   printf("  view create+traverse:    %8.1f ns/op  (%d iters, stbi_view_read)\n",
          (t1 - t0) * 1e9 / TRAV_ITERS, TRAV_ITERS);

   t0 = now_sec();
   for (i = 0; i < TRAV_ITERS; ++i) {
      int rw, rh;
      stbi_uc *ref = copy_crop(g_img, IMG_W, IMG_H, IMG_C, IMG_STRIDE,
                               rc.x, rc.y, rc.w, rc.h, STBI_VIEW_OOB_CLAMP, fill, &rw, &rh);
      int n = rw * rh * IMG_C, k;
      for (k = 0; k < n; ++k) sink += ref[k];
      free(ref);
   }
   t1 = now_sec();
   printf("  copy-crop (malloc+copy): %8.1f ns/op  (%d iters)\n",
          (t1 - t0) * 1e9 / TRAV_ITERS, TRAV_ITERS);

   if (sink == 0xdeadbeefUL) printf("  (sink %lu)\n", sink); // defeat dead-code elim
}

int main(void)
{
   static const stbi_uc fill[4] = { 255, 0, 255, 0 }; // configurable fill color: magenta
   static const struct rect_case interior[] = {
      { "interior", 10, 8, 40, 30 },
   };
   static const struct rect_case crossing[] = {
      { "cross right+bottom",  80, 50, 40, 30 },
      { "cross left+top",      -6, -4, 26, 14 },
   };
   static const struct rect_case outside[] = {
      { "outside right/bottom", 200, 200, 10, 10 },
      { "outside left",         -50,   0, 10, 10 },
   };
   static const struct rect_case zeroarea[] = {
      { "zero width",   5, 5,  0, 10 },
      { "zero height",  5, 5, 10,  0 },
      { "negative w/h", 5, 5, -3, -7 },
   };
   long allocs_before;

   printf("=== stbi_view sub-image view tests ===\n");
   printf("source image: %dx%dx%d, stride=%d bytes (aligned %d), buffer=%zu bytes\n\n",
          IMG_W, IMG_H, IMG_C, IMG_STRIDE, ALIGN, sizeof(g_img));
   fill_pattern();

   // ---- decode unchanged: PNG roundtrip through stbi_load must be pixel-exact
   {
      int png_len = 0, dx = 0, dy = 0, dc = 0, x, y, bad = 0;
      unsigned char *png = stbi_write_png_to_mem(g_img, IMG_STRIDE, IMG_W, IMG_H, IMG_C, &png_len);
      CHECK(png != NULL, "PNG encode failed");
      allocs_before = g_allocs;
      {
         stbi_uc *dec = stbi_load_from_memory(png, png_len, &dx, &dy, &dc, IMG_C);
         printf("=== decode check (stbi_load_from_memory, PNG %d bytes) ===\n", png_len);
         CHECK(dec != NULL, "decode failed");
         CHECK(dx == IMG_W && dy == IMG_H && dc == IMG_C, "decoded dimensions changed");
         if (dec) {
            for (y = 0; y < IMG_H; ++y)
               for (x = 0; x < IMG_W; ++x) {
                  const stbi_uc *a = dec + ((size_t) y * IMG_W + x) * IMG_C;
                  const stbi_uc *b = g_img + (size_t) y * IMG_STRIDE + (size_t) x * IMG_C;
                  if (memcmp(a, b, IMG_C) != 0) ++bad;
               }
            printf("  decoded %dx%dx%d, pixel mismatches vs source: %d\n", dx, dy, dc, bad);
            CHECK(bad == 0, "decoded pixels differ from original");
            printf("  stbi_load allocations during decode: %ld\n", g_allocs - allocs_before);
            stbi_image_free(dec);
         }
      }
      STBIW_FREE(png);
      printf("\n");
   }

   // ---- zero-allocation evidence for the whole view phase
   allocs_before = g_allocs;

   run_rect_group("group 1: interior rectangle", interior, 1, fill, 1);
   run_rect_group("group 2: boundary-crossing rectangles", crossing, 2, fill, 1);
   run_rect_group("group 3: fully-outside rectangles (must error)", outside, 2, fill, 0);
   run_rect_group("group 4: zero-area rectangles (must error)", zeroarea, 3, fill, 0);
   run_alignment_cases();
   run_timing(fill);

   printf("\nview-phase allocations (create+read+traverse, all groups): %ld (expect 0)\n",
          g_allocs - allocs_before);
   CHECK(g_allocs == allocs_before, "view API allocated memory");

   printf("\n%s (%d failures)\n", g_failures == 0 ? "ALL PASS" : "FAILURES", g_failures);
   return g_failures == 0 ? 0 : 1;
}
