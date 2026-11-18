/* test_bilateral_grid.c - acceptance test for stb_bilateral.h
 *
 * Generates four small synthetic 256x256 grayscale images, runs the exact
 * bilateral filter (reference) and the bilateral-grid approximation with
 * several grid sizes, and reports PSNR, wall time and scratch memory.
 *
 * Build: see run_test.sh in this directory. No third-party libraries beyond
 * the stb headers already in the repository are used.
 */

#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ---- memory instrumentation: wrap the header's allocator ------------- */
static long long g_bytes_live = 0;
static long long g_bytes_peak = 0;

static void *track_malloc(size_t sz)
{
   void *p = malloc(sz + sizeof(size_t));
   if (!p) return NULL;
   memcpy(p, &sz, sizeof(size_t));
   g_bytes_live += (long long) sz;
   if (g_bytes_live > g_bytes_peak) g_bytes_peak = g_bytes_live;
   return (char *) p + sizeof(size_t);
}

static void track_free(void *p)
{
   size_t sz;
   if (!p) return;
   p = (char *) p - sizeof(size_t);
   memcpy(&sz, p, sizeof(size_t));
   g_bytes_live -= (long long) sz;
   free(p);
}

#define STBBG_MALLOC(sz) track_malloc(sz)
#define STBBG_FREE(p)    track_free(p)

#define STB_BILATERAL_IMPLEMENTATION
#include "../../stb_bilateral.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../stb_image_write.h"

#define W 256
#define H 256
#define NPIX (W * H)
#define SIGMA_S 8.0f
#define SIGMA_R 24.0f

/* deterministic LCG so the "noise" images are reproducible */
static unsigned long rng_state = 0x12345678UL;
static unsigned int rng_next(void)
{
   rng_state = rng_state * 6364136223846793005UL + 1442695040888963407UL;
   return (unsigned int)(rng_state >> 33);
}

static float rng_uniform(void)
{
   return (float) rng_next() / 4294967296.0f;
}

/* gaussian noise via Box-Muller */
static float rng_gauss(void)
{
   float u1 = rng_uniform() + 1e-9f;
   float u2 = rng_uniform();
   return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

static unsigned char clamp8(int v)
{
   if (v < 0) return 0;
   if (v > 255) return 255;
   return (unsigned char) v;
}/* four test patterns */
static void make_gradient(unsigned char *img)
{
   int i, j;
   for (j = 0; j < H; j++)
      for (i = 0; i < W; i++)
         img[j * W + i] = clamp8((int)((float) i * 255.0f / (W - 1)
                                       + 20.0f * sinf((float) j * 0.11f)));
}

static void make_steps(unsigned char *img)
{
   int i, j;
   for (j = 0; j < H; j++)
      for (i = 0; i < W; i++) {
         int v = (i / 32 + j / 64) % 2 ? 200 : 40;
         img[j * W + i] = (unsigned char) v;
      }
}

static void make_noise(unsigned char *img)
{
   int i, j;
   rng_state = 0xdeadbeefUL;
   for (j = 0; j < H; j++)
      for (i = 0; i < W; i++)
         img[j * W + i] = clamp8((int)(128 + 35.0f * rng_gauss()
                                       + 18.0f * sinf((float) i * 0.4f)));
}

static void make_mixed(unsigned char *img)
{
   int i, j;
   rng_state = 0xcafef00dUL;
   for (j = 0; j < H; j++)
      for (i = 0; i < W; i++) {
         float dx = (float) i - W * 0.5f, dy = (float) j - H * 0.5f;
         int disk = (dx * dx + dy * dy) < 55.0f * 55.0f;
         float ramp = (float) i * 200.0f / (W - 1);
         float v = disk ? 235.0f : ramp;
         v += 14.0f * rng_gauss();
         if ((i / 64 + j / 64) % 3 == 0) v = 255.0f - v; /* hard edges */
         img[j * W + i] = clamp8((int) v);
      }
}

static double psnr_db(const unsigned char *a, const unsigned char *b)
{
   double sse = 0.0;
   size_t k;
   for (k = 0; k < NPIX; k++) {
      double d = (double) a[k] - (double) b[k];
      sse += d * d;
   }
   if (sse <= 0.0) return 99.0;
   return 10.0 * log10((double) NPIX * 255.0 * 255.0 / sse);
}

/* Independent separable spatial Gaussian (clamp-to-edge), used as the
 * mathematical reference for the degenerate gx=gy=1 case, which must
 * reduce to a pure range-independent spatial Gaussian blur. Written from
 * scratch (not using stb_bilateral internals) so it is a true cross-check. */
static void spatial_gaussian_ref(const unsigned char *in, unsigned char *out,
                                 float sigma)
{
   int radius = (int) ceil(3.0 * sigma);
   int k, i, j;
   float *kern = malloc(sizeof(float) * (size_t)(2 * radius + 1));
   float *tmp = malloc(sizeof(float) * NPIX);
   double s = 0.0;
   for (k = -radius; k <= radius; k++) {
      kern[k + radius] = (float) exp(-0.5 * k * k / (sigma * sigma));
      s += kern[k + radius];
   }
   for (k = 0; k < 2 * radius + 1; k++) kern[k] /= (float) s;
   /* horizontal */
   for (j = 0; j < H; j++)
      for (i = 0; i < W; i++) {
         double acc = 0.0;
         for (k = -radius; k <= radius; k++) {
            int xx = i + k;
            if (xx < 0) xx = 0;
            if (xx > W - 1) xx = W - 1;
            acc += in[j * W + xx] * kern[k + radius];
         }
         tmp[j * W + i] = (float) acc;
      }
   /* vertical */
   for (j = 0; j < H; j++)
      for (i = 0; i < W; i++) {
         double acc = 0.0;
         for (k = -radius; k <= radius; k++) {
            int yy = j + k;
            if (yy < 0) yy = 0;
            if (yy > H - 1) yy = H - 1;
            acc += tmp[yy * W + i] * kern[k + radius];
         }
         out[j * W + i] = clamp8((int) floor(acc + 0.5));
      }
   free(tmp);
   free(kern);
}

static double now_sec(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double time_exact(const unsigned char *in, unsigned char *out)
{
   double t0 = now_sec();
   int rc = stbbg_bilateral_exact_gray8(W, H, in, out, SIGMA_S, SIGMA_R);
   double t1 = now_sec();
   if (rc) { fprintf(stderr, "exact filter failed\n"); exit(1); }
   return t1 - t0;
}

typedef struct
{
   const char *name;
   int gx, gy, gz, radius;
   int kind;
} grid_cfg;

int main(void)
{
   static unsigned char img[NPIX], ref[NPIX], out[NPIX];
   static const struct { const char *name; void (*make)(unsigned char *); }
      patterns[4] = {
         { "gradient", make_gradient },
         { "steps",    make_steps    },
         { "noise",    make_noise    },
         { "mixed",    make_mixed    },
      };
   /* three quality/speed tradeoff groups plus two degenerate configs.
    * kind 0 = normal tradeoff, 1 = pure spatial blur (gx=gy=1),
    * 2 = fully collapsed grid (single cell -> image-wide mean). */
   grid_cfg cfgs[5] = {
      { "coarse  gx=16  gy=16  gz=16",  16,  16, 16,  0, 0 },
      { "medium  gx=32  gy=32  gz=16",  32,  32, 16,  0, 0 },
      { "fine    gx=64  gy=64  gz=32",  64,  64, 32,  0, 0 },
      { "spatial-only gx=1 gy=1 gz=16",  1,   1, 16,  0, 1 },
      { "collapsed   gx=1 gy=1 gz=1",    1,   1,  1,  0, 2 },
   };
   int p, c, rep;
   long long exact_bytes;
   double t_exact;

   printf("stb_bilateral acceptance test  %dx%d images  "
          "sigma_spatial=%.1f sigma_range=%.1f\n", W, H, SIGMA_S, SIGMA_R);
   printf("exact window radius = ceil(3*sigma_s) = %d px\n\n",
          (int) ceil(3.0 * SIGMA_S));

   for (p = 0; p < 4; p++) {
      char path[256];
      patterns[p].make(img);
      snprintf(path, sizeof(path),
               "artifacts_bilateral/input_%s.png", patterns[p].name);
      stbi_write_png(path, W, H, 1, img, W);

      /* reference: exact bilateral, timed twice, take the best (warm cache) */
      t_exact = 1e30;
      g_bytes_peak = 0;
      for (rep = 0; rep < 2; rep++) {
         double t = time_exact(img, out);
         if (t < t_exact) t_exact = t;
      }
      exact_bytes = g_bytes_peak;
      memcpy(ref, out, NPIX);
      snprintf(path, sizeof(path),
               "artifacts_bilateral/ref_%s.png", patterns[p].name);
      stbi_write_png(path, W, H, 1, ref, W);

      printf("=== image %d: %s ===\n", p + 1, patterns[p].name);
      printf("  [exact ] time=%7.1f ms  scratch=%8lld B  (reference)\n",
             t_exact * 1e3, exact_bytes);

      for (c = 0; c < 5; c++) {
         stbbg_grid_params gp;
         double t_best = 1e30, psnr;
         long long reported = 0;
         gp.grid_w = cfgs[c].gx;
         gp.grid_h = cfgs[c].gy;
         gp.luma_levels = cfgs[c].gz;
         gp.sigma_spatial = SIGMA_S;
         gp.sigma_range = SIGMA_R;
         gp.blur_radius = cfgs[c].radius;

         g_bytes_peak = 0;
         for (rep = 0; rep < 5; rep++) {
            double t0, t1;
            t0 = now_sec();
            if (stbbg_bilateral_grid_gray8(W, H, img, out, &gp,
                                           &reported) != 0) {
               fprintf(stderr, "grid filter failed: %s\n", cfgs[c].name);
               return 1;
            }
            t1 = now_sec();
            if (t1 - t0 < t_best) t_best = t1 - t0;
         }
         psnr = psnr_db(ref, out);
         printf("  [grid %-31s] time=%7.2f ms  speedup=%6.2fx  "
                "PSNR=%6.2f dB  scratch=%8lld B  (reported %lld B)\n",
                cfgs[c].name, t_best * 1e3, t_exact / t_best, psnr,
                g_bytes_peak, reported);

         /* save the medium and degenerate outputs for visual inspection */
         if (c == 1) {
            snprintf(path, sizeof(path),
                     "artifacts_bilateral/grid_%s.png", patterns[p].name);
            stbi_write_png(path, W, H, 1, out, W);
         }
         if (cfgs[c].kind == 1) {
            snprintf(path, sizeof(path),
                     "artifacts_bilateral/spatial_%s.png", patterns[p].name);
            stbi_write_png(path, W, H, 1, out, W);
         }
         if (cfgs[c].kind == 2) {
            snprintf(path, sizeof(path),
                     "artifacts_bilateral/collapsed_%s.png", patterns[p].name);
            stbi_write_png(path, W, H, 1, out, W);
         }

         if (cfgs[c].kind == 0) {
            /* acceptance: PSNR >= 38 dB and >= 5x speedup */
            if (psnr < 38.0)
               printf("    FAIL: PSNR %.2f < 38 dB\n", psnr);
            if (t_exact / t_best < 5.0)
               printf("    FAIL: speedup %.2fx < 5x\n", t_exact / t_best);
         } else if (cfgs[c].kind == 1) {
            /* gx=gy=1 must equal a pure spatial Gaussian (range term
             * cannot matter); compare against the independent reference */
            static unsigned char expect[NPIX];
            double q;
            spatial_gaussian_ref(img, expect, SIGMA_S);
            q = psnr_db(expect, out);
            printf("    degenerate spatial-only: PSNR vs spatial-Gaussian "
                   "reference = %.2f dB", q);
            if (q < 45.0)
               printf("  FAIL (expected ~exact match)");
            printf("\n");
         } else {
            /* fully collapsed grid: by contract this is a pure spatial
             * Gaussian (range has no room to act). It must match the
             * independent spatial-Gaussian reference with no zero-division,
             * and produce finite, in-range results everywhere. */
            static unsigned char expect[NPIX];
            double q;
            size_t k;
            int bounded = 1;
            spatial_gaussian_ref(img, expect, SIGMA_S);
            q = psnr_db(expect, out);
            /* unsigned char is inherently 0..255; assert a bounded,
             * non-garbage delta against the spatial-Gaussian reference */
            for (k = 0; k < NPIX; k++)
               if (abs((int) out[k] - (int) expect[k]) > 20) {
                  bounded = 0; break;
               }
            printf("    collapsed single-cell: PSNR vs spatial-Gaussian "
                   "reference = %.2f dB, bounded=%d %s\n",
                   q, bounded, (q >= 45.0 && bounded) ? "OK" : "FAIL");
            if (q < 45.0 || !bounded) return 1;
         }
      }
      printf("\n");
   }

   /* ---- explicit edge-case checks ------------------------------------ */
   {
      stbbg_grid_params gp;
      unsigned char flat[64], fout[64];
      int k, rc;
      for (k = 0; k < 64; k++) flat[k] = 100;
      memset(fout, 0, sizeof(fout));
      gp.grid_w = 1; gp.grid_h = 1; gp.luma_levels = 1;
      gp.sigma_spatial = SIGMA_S; gp.sigma_range = SIGMA_R;
      gp.blur_radius = 0;
      rc = stbbg_bilateral_grid_gray8(8, 8, flat, fout, &gp, NULL);
      if (rc != 0 || fout[0] != 100 || fout[63] != 100) {
         printf("EDGE FAIL: gx=gy=gz=1 on flat image rc=%d v0=%d v63=%d\n",
                rc, fout[0], fout[63]);
         return 1;
      }
      printf("edge case gx=gy=gz=1 flat image: output=%d (expected 100) OK\n",
             fout[0]);

      /* bad arguments must fail cleanly rather than crash */
      gp.grid_w = 0;
      if (stbbg_bilateral_grid_gray8(8, 8, flat, fout, &gp, NULL) != -1) {
         printf("EDGE FAIL: grid_w=0 accepted\n");
         return 1;
      }
      if (stbbg_bilateral_exact_gray8(0, 8, flat, fout, 1, 1) != -1) {
         printf("EDGE FAIL: w=0 accepted\n");
         return 1;
      }
      printf("edge case invalid arguments rejected OK\n");
      if (g_bytes_live != 0) {
         printf("EDGE FAIL: memory leak, live=%lld\n", g_bytes_live);
         return 1;
      }
      printf("memory instrumentation: all allocations freed OK\n");
   }
   printf("\nAll checks finished.\n");
   return 0;
}
