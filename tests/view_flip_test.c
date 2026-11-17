// view_flip_test.c - tests for the stbi_view flipped-view API in stb_image.h
//
// Covers, for each of the 4 view combinations (identity, row-reversed,
// column-reversed, row+column reversed):
//   - access consistency vs a copy-flip reference, pixel by pixel
//   - allocation count during view creation (must be 0; views alias the
//     decode buffer)
//   - traversal timing vs the copy-flip alternative
//   - out-of-bounds access behavior (must return 0)
// Plus:
//   - decode output unchanged, pixel by pixel (PNG roundtrip of a known
//     pattern + identity view matches the raw decode buffer)
//   - view invalidation after stbi_image_free(): tracked-allocator registry
//     check, and (in ASan builds) a forked child that dereferences the
//     dangling view and must be killed by AddressSanitizer.
//
// Build/run via run_view_flip_test.sh (not wired into the repo Makefile).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__SANITIZE_ADDRESS__)
#define VIEW_TEST_ASAN 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define VIEW_TEST_ASAN 1
#endif
#endif
#ifndef VIEW_TEST_ASAN
#define VIEW_TEST_ASAN 0
#endif

#if VIEW_TEST_ASAN
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#endif

// ---------------------------------------------------------------------------
// tracked allocator: counts calls and keeps a registry of live blocks so the
// test can prove (a) view creation allocates nothing and (b) a view's owner
// buffer is no longer live after stbi_image_free().
// ---------------------------------------------------------------------------

static long trk_alloc_calls = 0;
static long trk_realloc_calls = 0;
static long trk_free_calls  = 0;

#define TRK_MAX_LIVE 8192
static void *trk_live[TRK_MAX_LIVE];
static int   trk_live_n = 0;

static void trk_add(void *p)
{
   if (p && trk_live_n < TRK_MAX_LIVE)
      trk_live[trk_live_n++] = p;
}

static void trk_remove(void *p)
{
   int i;
   for (i = 0; i < trk_live_n; ++i)
      if (trk_live[i] == p) {
         trk_live[i] = trk_live[--trk_live_n];
         return;
      }
}

static int trk_contains(const void *p)
{
   int i;
   for (i = 0; i < trk_live_n; ++i)
      if (trk_live[i] == p)
         return 1;
   return 0;
}

static void *trk_malloc(size_t n)
{
   void *p = malloc(n);
   trk_alloc_calls++;
   trk_add(p);
   return p;
}

static void *trk_realloc(void *p, size_t n)
{
   void *q;
   trk_realloc_calls++;
   trk_remove(p);
   q = realloc(p, n);
   trk_add(q);
   return q;
}

static void trk_free(void *p)
{
   if (p)
      trk_free_calls++;
   trk_remove(p);
   free(p);
}

#define STBI_MALLOC(sz)        trk_malloc(sz)
#define STBI_REALLOC(p, sz)    trk_realloc(p, sz)
#define STBI_FREE(p)           trk_free(p)

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// ---------------------------------------------------------------------------

static int g_pass = 0, g_fail = 0;

static void check(int cond, const char *what)
{
   if (cond) {
      g_pass++;
      printf("    PASS  %s\n", what);
   } else {
      g_fail++;
      printf("    FAIL  %s\n", what);
   }
}

static double now_sec(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
}

// deterministic pattern, asymmetric in x and y so flips are detectable
static unsigned char pattern_px(int x, int y, int c)
{
   return (unsigned char) (x * 17 + y * 31 + c * 47 + x * y * 5 + 3);
}

// copy-flip reference: allocates a new buffer and copies pixels flipped
static unsigned char *copy_flip(const unsigned char *img, int w, int h, int comp,
                                int flip_x, int flip_y)
{
   unsigned char *out = (unsigned char *) malloc((size_t) w * h * comp);
   int x, y, c;
   for (y = 0; y < h; ++y)
      for (x = 0; x < w; ++x) {
         int sx = flip_x ? w - 1 - x : x;
         int sy = flip_y ? h - 1 - y : y;
         for (c = 0; c < comp; ++c)
            out[((size_t) y * w + x) * comp + c] =
               img[((size_t) sy * w + sx) * comp + c];
      }
   return out;
}

typedef stbi_view (*view_make_fn)(stbi_uc *, int, int, int);

typedef struct {
   const char  *name;
   int          flip_x, flip_y;
   view_make_fn make;
} view_group;

static volatile unsigned int g_sink; // prevents the timing loops from being optimized away

// traverse the whole view once, summing all bytes; walks the public
// data/stride_x/stride_y fields directly, the intended hot-loop usage.
// Rows whose pixels are contiguous (stride_x == comp: identity and
// row-reversed views) are summed as flat runs; only column-reversed
// views pay for truly strided access.
static unsigned int traverse_view(const stbi_view *v)
{
   unsigned int s = 0;
   int x, y, c;
   const stbi_uc *row = v->data;
   if (v->stride_x == v->comp) {
      size_t rowbytes = (size_t) v->x * v->comp, i;
      for (y = 0; y < v->y; ++y, row += v->stride_y)
         for (i = 0; i < rowbytes; ++i)
            s += row[i];
   } else {
      for (y = 0; y < v->y; ++y, row += v->stride_y) {
         const stbi_uc *p = row;
         for (x = 0; x < v->x; ++x, p += v->stride_x)
            for (c = 0; c < v->comp; ++c)
               s += p[c];
      }
   }
   return s;
}

static unsigned int traverse_buf(const unsigned char *b, int w, int h, int comp)
{
   unsigned int s = 0;
   size_t i, n = (size_t) w * h * comp;
   for (i = 0; i < n; ++i)
      s += b[i];
   return s;
}

static void run_group(const view_group *g, stbi_uc *img, int w, int h, int comp,
                      int timing_iters)
{
   stbi_view v;
   unsigned char *ref;
   long a0, r0;
   int x, y, c, mismatches = 0;
   char msg[160];
   double t0, t_view, t_copy;

   printf("  group: %s\n", g->name);

   // --- allocation count: snapshot the tracked-allocator counters around
   //     view creation; the delta must be exactly 0
   a0 = trk_alloc_calls; r0 = trk_realloc_calls;
   v = g->make(img, w, h, comp);
   snprintf(msg, sizeof msg, "allocations during view creation: %ld (malloc) + %ld (realloc)",
            trk_alloc_calls - a0, trk_realloc_calls - r0);
   check(trk_alloc_calls - a0 == 0 && trk_realloc_calls - r0 == 0, msg);

   // --- memory sharing evidence: view base pointer lies inside the decode
   //     buffer, and a write to the decode buffer is visible through the view
   snprintf(msg, sizeof msg, "view base %p inside decode buffer [%p, %p)",
            (void *) v.data, (void *) img, (void *) (img + (size_t) w * h * comp));
   check(v.data >= img && v.data < img + (size_t) w * h * comp, msg);
   printf("    info  strides: stride_x=%td stride_y=%td (bytes)\n",
          v.stride_x, v.stride_y);
   check((g->flip_x ? v.stride_x < 0 : v.stride_x > 0) &&
         (g->flip_y ? v.stride_y < 0 : v.stride_y > 0),
         "stride signs match flip directions");
   {
      int wx = w > 1 ? 1 : 0, wy = h > 1 ? 1 : 0, wc = 0;
      int px = g->flip_x ? w - 1 - wx : wx;
      int py = g->flip_y ? h - 1 - wy : wy;
      unsigned char old = img[((size_t) py * w + px) * comp + wc];
      img[((size_t) py * w + px) * comp + wc] = (unsigned char) (old ^ 0x5A);
      check(stbi_view_at(&v, wx, wy, wc) == (unsigned char) (old ^ 0x5A),
            "write to decode buffer is visible through the view (shared memory)");
      img[((size_t) py * w + px) * comp + wc] = old;
   }

   // --- access consistency vs copy-flip reference, pixel by pixel
   ref = copy_flip(img, w, h, comp, g->flip_x, g->flip_y);
   for (y = 0; y < h; ++y)
      for (x = 0; x < w; ++x)
         for (c = 0; c < comp; ++c)
            if (stbi_view_at(&v, x, y, c) != ref[((size_t) y * w + x) * comp + c])
               mismatches++;
   snprintf(msg, sizeof msg, "pixel-by-pixel match with copy-flip reference (%d/%d bytes differ)",
            mismatches, w * h * comp);
   check(mismatches == 0, msg);

   // --- out-of-bounds behavior: defined, returns 0
   check(stbi_view_at(&v, -1, 0, 0) == 0 &&
         stbi_view_at(&v, 0, -1, 0) == 0 &&
         stbi_view_at(&v, 0, 0, -1) == 0 &&
         stbi_view_at(&v, w, 0, 0) == 0 &&
         stbi_view_at(&v, 0, h, 0) == 0 &&
         stbi_view_at(&v, 0, 0, comp) == 0 &&
         stbi_view_at(&v, w + 100, h + 100, comp + 100) == 0,
         "out-of-bounds access returns 0 (x/y/c, negative and past-end)");

   // --- timing: (view create + traverse) vs (malloc + flip-copy + traverse + free)
   {
      int i;
      t0 = now_sec();
      for (i = 0; i < timing_iters; ++i) {
         stbi_view tv = g->make(img, w, h, comp);
         g_sink += traverse_view(&tv);
      }
      t_view = now_sec() - t0;

      t0 = now_sec();
      for (i = 0; i < timing_iters; ++i) {
         unsigned char *cp = copy_flip(img, w, h, comp, g->flip_x, g->flip_y);
         g_sink += traverse_buf(cp, w, h, comp);
         free(cp);
      }
      t_copy = now_sec() - t0;

      printf("    info  timing over %d iters (%dx%dx%d): view %.3f ms, copy-flip %.3f ms (%.1fx)\n",
             timing_iters, w, h, comp, t_view * 1e3, t_copy * 1e3,
             t_copy > 0 ? t_copy / (t_view > 0 ? t_view : 1e-12) : 0.0);
   }

   free(ref);
}

int main(void)
{
   static const view_group groups[4] = {
      { "identity (no flip)",   0, 0, stbi_view_make      },
      { "row-reversed",         0, 1, stbi_view_flip_rows },
      { "column-reversed",      1, 0, stbi_view_flip_cols },
      { "row+column reversed",  1, 1, stbi_view_flip_both },
   };
   const int W = 9, H = 7, COMP = 4;
   unsigned char *src, *png, *img;
   int png_len = 0, w = 0, h = 0, comp = 0;
   int x, y, c, i, mismatches;
   stbi_view ident;
   char msg[160];

   printf("== stbi_view flipped-view tests ==\n");
#if VIEW_TEST_ASAN
   printf("   (AddressSanitizer build)\n");
#endif

   // --- build a small sample in memory and decode it for real
   src = (unsigned char *) malloc((size_t) W * H * COMP);
   for (y = 0; y < H; ++y)
      for (x = 0; x < W; ++x)
         for (c = 0; c < COMP; ++c)
            src[((size_t) y * W + x) * COMP + c] = pattern_px(x, y, c);
   png = stbi_write_png_to_mem(src, 0, W, H, COMP, &png_len);
   if (!png) { printf("FATAL: png encode failed\n"); return 1; }
   img = stbi_load_from_memory(png, png_len, &w, &h, &comp, COMP);
   STBIW_FREE(png);
   if (!img) { printf("FATAL: decode failed: %s\n", stbi_failure_reason()); return 1; }
   printf("  sample: %dx%dx%d PNG, decoded in-process\n", w, h, COMP);

   // --- original decode unchanged, pixel by pixel
   mismatches = 0;
   for (i = 0; i < W * H * COMP; ++i)
      if (img[i] != src[i])
         mismatches++;
   snprintf(msg, sizeof msg, "decode output identical to source pattern (%d/%d bytes differ)",
            mismatches, W * H * COMP);
   check(mismatches == 0, msg);

   // --- identity view matches the raw decode buffer pixel by pixel
   ident = stbi_view_make(img, w, h, COMP);
   mismatches = 0;
   for (y = 0; y < h; ++y)
      for (x = 0; x < w; ++x)
         for (c = 0; c < COMP; ++c)
            if (stbi_view_at(&ident, x, y, c) != img[((size_t) y * w + x) * COMP + c])
               mismatches++;
   check(mismatches == 0, "identity view matches raw decode buffer pixel by pixel");

   // --- the four view combinations
   for (i = 0; i < 4; ++i)
      run_group(&groups[i], img, w, h, COMP, 200000);

   // --- a second, larger sample to show how the timing gap scales
   {
      const int W2 = 64, H2 = 48;
      unsigned char *src2 = (unsigned char *) malloc((size_t) W2 * H2 * COMP);
      unsigned char *png2, *img2;
      int png2_len = 0, w2 = 0, h2 = 0, c2 = 0;
      for (y = 0; y < H2; ++y)
         for (x = 0; x < W2; ++x)
            for (c = 0; c < COMP; ++c)
               src2[((size_t) y * W2 + x) * COMP + c] = pattern_px(x, y, c);
      png2 = stbi_write_png_to_mem(src2, 0, W2, H2, COMP, &png2_len);
      img2 = stbi_load_from_memory(png2, png2_len, &w2, &h2, &c2, COMP);
      STBIW_FREE(png2);
      if (!img2) { printf("FATAL: decode of second sample failed\n"); return 1; }
      printf("  larger sample: %dx%dx%d\n", w2, h2, COMP);
      for (i = 0; i < 4; ++i)
         run_group(&groups[i], img2, w2, h2, COMP, 20000);
      stbi_image_free(img2);
      free(src2);
   }

   // --- invalidation after the decode buffer is freed
   printf("  invalidation after stbi_image_free():\n");
   {
      stbi_view v = stbi_view_flip_both(img, w, h, COMP);
      check(trk_contains(stbi_view_owner(&v)),
            "before free: view owner is a live tracked allocation");
      stbi_image_free(img);
      check(!trk_contains(stbi_view_owner(&v)),
            "after free: view owner no longer live (tracked-allocator registry detects invalidation)");
#if VIEW_TEST_ASAN
      {
         pid_t pid = fork();
         if (pid == 0) {
            // child: dereference the dangling view; ASan must kill us
            volatile unsigned int s = stbi_view_at(&v, 0, 0, 0);
            (void) s;
            _exit(42); // survived: dangling access NOT caught
         }
         if (pid > 0) {
            int status = 0;
            waitpid(pid, &status, 0);
            check(!(WIFEXITED(status) && WEXITSTATUS(status) == 42),
                  "after free: dereferencing the view is caught by AddressSanitizer (child killed)");
         }
      }
#else
      printf("    info  (non-ASan build: dangling-dereference check skipped, run the ASan build)\n");
#endif
   }

   free(src);
   printf("== %d checks passed, %d failed ==\n", g_pass, g_fail);
   return g_fail ? 1 : 0;
}
