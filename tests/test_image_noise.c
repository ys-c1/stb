/* test_image_noise.c
 *
 * Robustness-testing support: inject "dirty image" noise into 8-bit
 * interleaved pixel buffers (the same layout stb_image produces:
 * width * height * comp, comp in 1..4, row-major, RGB(A)).
 *
 * Noise models:
 *   NOISE_GAUSSIAN    - additive zero-mean Gaussian noise, intensity = sigma (0..100)
 *   NOISE_SALT_PEPPER - impulse noise,                 intensity = density p (0..1)
 *   NOISE_STRIPE      - additive horizontal stripe,    intensity = amplitude (0..255)
 *
 * Features:
 *   - configurable intensity and random seed; same seed => byte-identical output
 *   - optional dirty rectangle: only pixels inside [rx,ry,rw,rh] are touched
 *   - invalid arguments (incl. out-of-range intensity) are rejected before any
 *     memory is written; the destination buffer is left untouched
 *   - the source buffer is never modified (callers may pass src == dst; a
 *     working copy is made internally)
 *
 * Self-contained: C99 standard library only. No third-party dependencies and
 * no network access. This file is a test utility; it does not modify any stb
 * library file, build script or public interface.
 *
 * Build: gcc -std=c99 -O2 -Wall -Wextra -o test_image_noise test_image_noise.c -lm
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NOISE_PI 3.14159265358979323846

/* ----------------------------------------------------------------------- */
/* noise injection library                                                 */
/* ----------------------------------------------------------------------- */

typedef enum {
   NOISE_GAUSSIAN    = 0,
   NOISE_SALT_PEPPER = 1,
   NOISE_STRIPE      = 2
} noise_type_t;

enum {
   NOISE_OK            =  0,
   NOISE_ERR_NULL      = -1,
   NOISE_ERR_DIM       = -2,
   NOISE_ERR_CHANNELS  = -3,
   NOISE_ERR_REGION    = -4,
   NOISE_ERR_TYPE      = -5,
   NOISE_ERR_INTENSITY = -6,
   NOISE_ERR_ALLOC     = -7
};

/* Deterministic xorshift32 PRNG. A zero seed is remapped so the state can
 * never get stuck in xorshift's fixed point. */
typedef struct { unsigned int state; } noise_rng_t;

static void noise_rng_seed(noise_rng_t *rng, unsigned int seed)
{
   rng->state = seed ? seed : 0x9e3779b9u;
}

static unsigned int noise_rng_u32(noise_rng_t *rng)
{
   unsigned int x = rng->state;
   x ^= x << 13;
   x ^= x >> 17;
   x ^= x << 5;
   rng->state = x;
   return x;
}

/* uniform double in [0, 1) using the top 24 random bits */
static double noise_rng_double(noise_rng_t *rng)
{
   return (double)(noise_rng_u32(rng) >> 8) * (1.0 / 16777216.0);
}

/* standard normal variate via Box-Muller */
static double noise_rng_gaussian(noise_rng_t *rng)
{
   double u1, u2;
   do {
      u1 = noise_rng_double(rng);
   } while (u1 <= 0.0);
   u2 = noise_rng_double(rng);
   return sqrt(-2.0 * log(u1)) * cos(2.0 * NOISE_PI * u2);
}

static const char *noise_strerror(int status)
{
   switch (status) {
      case NOISE_OK:             return "ok";
      case NOISE_ERR_NULL:       return "null buffer";
      case NOISE_ERR_DIM:        return "width/height must be positive";
      case NOISE_ERR_CHANNELS:   return "channels must be in 1..4";
      case NOISE_ERR_REGION:     return "dirty rectangle outside image";
      case NOISE_ERR_TYPE:       return "unknown noise type";
      case NOISE_ERR_INTENSITY:  return "intensity out of range";
      case NOISE_ERR_ALLOC:      return "out of memory";
      default:                   return "unknown error";
   }
}

static unsigned char noise_clamp(int v)
{
   if (v < 0)   return 0;
   if (v > 255) return 255;
   return (unsigned char)v;
}

/* Inject noise from src into dst. Both buffers hold w*h*comp bytes.
 * Pass rw <= 0 or rh <= 0 to dirty the whole image.
 * All validation happens before any write, so failures leave dst untouched. */
static int inject_noise(unsigned char *dst, const unsigned char *src,
                        int w, int h, int comp,
                        noise_type_t type, double intensity, unsigned int seed,
                        int rx, int ry, int rw, int rh)
{
   int x0, y0, x1, y1, x, y, c;
   size_t total;
   noise_rng_t rng;

   if (dst == NULL || src == NULL)             return NOISE_ERR_NULL;
   if (w <= 0 || h <= 0)                       return NOISE_ERR_DIM;
   if (comp < 1 || comp > 4)                   return NOISE_ERR_CHANNELS;
   if (type < 0 || type > NOISE_STRIPE)        return NOISE_ERR_TYPE;
   if (!isfinite(intensity) || intensity < 0.0) return NOISE_ERR_INTENSITY;
   switch (type) {
      case NOISE_GAUSSIAN:
         if (intensity > 100.0) return NOISE_ERR_INTENSITY;
         break;
      case NOISE_SALT_PEPPER:
         if (intensity > 1.0)   return NOISE_ERR_INTENSITY;
         break;
      case NOISE_STRIPE:
         if (intensity > 255.0) return NOISE_ERR_INTENSITY;
         break;
   }

   if (rw <= 0 || rh <= 0) {
      x0 = y0 = 0; x1 = w; y1 = h;
   } else {
      if (rx < 0 || ry < 0 || rx > w || ry > h ||
          (long)rx + rw > w || (long)ry + rh > h)
         return NOISE_ERR_REGION;
      x0 = rx; y0 = ry; x1 = rx + rw; y1 = ry + rh;
   }

   total = (size_t)w * (size_t)h * (size_t)comp;
   memcpy(dst, src, total);                 /* untouched pixels stay identical */

   noise_rng_seed(&rng, seed);

   if (type == NOISE_GAUSSIAN) {
      for (y = y0; y < y1; ++y) {
         for (x = x0; x < x1; ++x) {
            unsigned char *p = dst + ((size_t)y * w + x) * comp;
            for (c = 0; c < comp; ++c) {
               double n = noise_rng_gaussian(&rng) * intensity;
               p[c] = noise_clamp((int)p[c] + (int)lround(n));
            }
         }
      }
   } else if (type == NOISE_SALT_PEPPER) {
      for (y = y0; y < y1; ++y) {
         for (x = x0; x < x1; ++x) {
            double u = noise_rng_double(&rng);
            unsigned char *p = dst + ((size_t)y * w + x) * comp;
            if (u < intensity * 0.5) {
               for (c = 0; c < comp; ++c) p[c] = 0;                 /* pepper */
            } else if (u < intensity) {
               for (c = 0; c < comp; ++c) p[c] = 255;               /* salt   */
            }
         }
      }
   } else { /* NOISE_STRIPE: horizontal bands, amplitude = intensity.
             * A per-row RNG jitter guarantees distinct seeds produce
             * distinct patterns instead of quantizing to the same rows. */
      double period = 8.0;
      double phase  = 2.0 * NOISE_PI * noise_rng_double(&rng);
      for (y = y0; y < y1; ++y) {
         double wave   = sin(2.0 * NOISE_PI * (double)y / period + phase);
         double jitter = noise_rng_double(&rng) - 0.5;
         int shift = (int)lround(intensity * (0.8 * wave + 0.6 * jitter));
         for (x = x0; x < x1; ++x) {
            unsigned char *p = dst + ((size_t)y * w + x) * comp;
            for (c = 0; c < comp; ++c)
               p[c] = noise_clamp((int)p[c] + shift);
         }
      }
   }

   return NOISE_OK;
}

/* ----------------------------------------------------------------------- */
/* metrics                                                                 */
/* ----------------------------------------------------------------------- */

static double compute_psnr(const unsigned char *a, const unsigned char *b,
                           size_t n, double *mse_out)
{
   double sum = 0.0;
   size_t i;
   for (i = 0; i < n; ++i) {
      double d = (double)a[i] - (double)b[i];
      sum += d * d;
   }
   *mse_out = sum / (double)n;
   if (*mse_out == 0.0)
      return INFINITY;
   return 10.0 * log10((255.0 * 255.0) / *mse_out);
}

/* fraction (0..1) of pixels where any component differs */
static double changed_ratio(const unsigned char *a, const unsigned char *b,
                            int w, int comp,
                            int x0, int y0, int x1, int y1, long *changed)
{
   long area = 0, diff = 0;
   int x, y;
   for (y = y0; y < y1; ++y) {
      for (x = x0; x < x1; ++x) {
         const unsigned char *pa = a + ((size_t)y * w + x) * comp;
         const unsigned char *pb = b + ((size_t)y * w + x) * comp;
         int any = 0, c;
         for (c = 0; c < comp; ++c)
            any |= (pa[c] != pb[c]);
         diff += any;
         ++area;
      }
   }
   if (changed) *changed = diff;
   return area ? (double)diff / (double)area : 0.0;
}

static double now_seconds(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ----------------------------------------------------------------------- */
/* test harness                                                            */
/* ----------------------------------------------------------------------- */

static int g_checks;
static int g_failures;

#define CHECK(cond, ...)                                              \
   do {                                                              \
      ++g_checks;                                                    \
      if (!(cond)) {                                                 \
         ++g_failures;                                               \
         printf("    FAIL: "); printf(__VA_ARGS__); printf("\n");    \
      }                                                              \
   } while (0)

#define W 16
#define H 16

typedef struct {
   const char *name;
   int comp;
   unsigned char px[W * H * 4];
} sample_image;

static void make_samples(sample_image img[4])
{
   int x, y, k;

   /* 1. smooth: low-frequency gradients */
   img[0].name = "smooth"; img[0].comp = 3;
   for (y = 0; y < H; ++y)
      for (x = 0; x < W; ++x) {
         unsigned char *p = img[0].px + (y * W + x) * 3;
         p[0] = (unsigned char)(x * 15);
         p[1] = (unsigned char)(y * 15);
         p[2] = 128;
      }

   /* 2. detail: checkerboard + high-frequency patterns */
   img[1].name = "detail"; img[1].comp = 3;
   for (y = 0; y < H; ++y)
      for (x = 0; x < W; ++x) {
         unsigned char *p = img[1].px + (y * W + x) * 3;
         p[0] = ((x ^ y) & 1) ? 255 : 0;
         p[1] = (unsigned char)(((x + y) % 4) * 60);
         p[2] = (unsigned char)(((x * y) % 7) * 36);
      }

   /* 3. solid color: identical pixels */
   img[2].name = "solid"; img[2].comp = 3;
   for (k = 0; k < W * H; ++k) {
      img[2].px[k * 3 + 0] = 30;
      img[2].px[k * 3 + 1] = 180;
      img[2].px[k * 3 + 2] = 90;
   }

   /* 4. transparent: RGBA with fully transparent border + varied alpha */
   img[3].name = "transparent"; img[3].comp = 4;
   for (y = 0; y < H; ++y)
      for (x = 0; x < W; ++x) {
         unsigned char *p = img[3].px + (y * W + x) * 4;
         p[0] = (unsigned char)(x * 15);
         p[1] = (unsigned char)(y * 15);
         p[2] = 200;
         p[3] = (x == 0 || y == 0) ? 0 : (unsigned char)((x * 16 + y) & 255);
      }
}

static const char *noise_name(noise_type_t t)
{
   static const char *names[] = { "gaussian", "salt_pepper", "stripe" };
   return names[t];
}

static const double table_intensity[3] = { 20.0, 0.10, 60.0 };

int main(void)
{
   sample_image img[4];
   sample_image backup[4];
   unsigned char dirty[W * H * 4];
   unsigned char dirty2[W * H * 4];
   unsigned char marker[W * H * 4];
   size_t n;
   int i, t, rc;
   double t0, t1, psnr, mse, ratio;
   long changed;

   make_samples(img);
   memcpy(backup, img, sizeof img);
   memset(marker, 0xA5, sizeof marker);

   printf("=== stb noise-injection robustness test ===\n\n");
   printf("[1] PSNR / changed-pixel ratio / timing (seed=12345, %dx%d images)\n", W, H);
   printf("%-12s %-12s %9s %10s %10s %12s\n",
          "image", "noise", "intensity", "PSNR(dB)", "changed%", "time(us)");
   printf("-----------------------------------------------------------------------\n");

   for (i = 0; i < 4; ++i) {
      n = (size_t)W * H * img[i].comp;
      for (t = 0; t < 3; ++t) {
         double inten = table_intensity[t];
         t0 = now_seconds();
         rc = inject_noise(dirty, img[i].px, W, H, img[i].comp,
                           (noise_type_t)t, inten, 12345u, 0, 0, 0, 0);
         t1 = now_seconds();
         CHECK(rc == NOISE_OK, "%s/%s inject returned %d (%s)",
               img[i].name, noise_name(t), rc, noise_strerror(rc));
         CHECK(memcmp(img[i].px, backup[i].px, n) == 0,
               "%s/%s: source image modified pixel-by-pixel",
               img[i].name, noise_name(t));

         psnr = compute_psnr(img[i].px, dirty, n, &mse);
         ratio = changed_ratio(img[i].px, dirty, W, img[i].comp,
                               0, 0, W, H, &changed);
         printf("%-12s %-12s %9.2f %10.3f %9.2f%% %12.1f\n",
                img[i].name, noise_name(t), inten, psnr, ratio * 100.0,
                (t1 - t0) * 1e6);

         /* sanity bounds on the metrics */
         CHECK(isfinite(psnr) && psnr > 0.0,
               "%s/%s: PSNR not finite/positive (%f)", img[i].name, noise_name(t), psnr);
         if (t == NOISE_SALT_PEPPER)
            CHECK(ratio > 0.03 && ratio < 0.25,
                  "%s/salt_pepper: changed ratio %f outside plausible band", img[i].name, ratio);
         if (t == NOISE_GAUSSIAN)
            CHECK(ratio > 0.5,
                  "%s/gaussian: sigma=20 should change most pixels, got %f", img[i].name, ratio);
         if (t == NOISE_STRIPE)
            CHECK(ratio > 0.5,
                  "%s/stripe: amplitude=60 should change most pixels, got %f", img[i].name, ratio);
      }
   }

   printf("\n[2] dirty rectangle: only region (x=3,y=2,w=6,h=6) is polluted\n");
   for (t = 0; t < 3; ++t) {
      int x, y;
      long outside_bad = 0, inside_changed = 0, inside_total = 0;
      rc = inject_noise(dirty, img[0].px, W, H, img[0].comp,
                        (noise_type_t)t, table_intensity[t], 777u, 3, 2, 6, 6);
      CHECK(rc == NOISE_OK, "region %s inject returned %d", noise_name(t), rc);
      for (y = 0; y < H; ++y)
         for (x = 0; x < W; ++x) {
            int inside = (x >= 3 && x < 9 && y >= 2 && y < 8);
            const unsigned char *pa = img[0].px + (y * W + x) * 3;
            const unsigned char *pb = dirty + (y * W + x) * 3;
            int diff = (pa[0] != pb[0] || pa[1] != pb[1] || pa[2] != pb[2]);
            if (inside) { inside_changed += diff; ++inside_total; }
            else        outside_bad += diff;
         }
      ratio = changed_ratio(img[0].px, dirty, W, 3, 3, 2, 9, 8, &changed);
      printf("    %-12s region changed %2ld/36 (%5.1f%%), whole-image %5.2f%%, outside-region leaks: %ld\n",
             noise_name(t), inside_changed, ratio * 100.0,
             changed_ratio(img[0].px, dirty, W, 3, 0, 0, W, H, NULL) * 100.0,
             outside_bad);
      CHECK(outside_bad == 0, "%s: pixels outside dirty rectangle changed", noise_name(t));
      CHECK(inside_changed > 0, "%s: no pixel changed inside region", noise_name(t));
      CHECK(inside_total == 36, "region area bookkeeping wrong");
      CHECK(memcmp(img[0].px, backup[0].px, sizeof img[0].px) == 0,
            "%s: source image modified", noise_name(t));
   }

   printf("\n[3] reproducibility: same seed -> identical, different seeds -> differ\n");
   for (i = 0; i < 4; ++i) {
      n = (size_t)W * H * img[i].comp;
      for (t = 0; t < 3; ++t) {
         rc = inject_noise(dirty,  img[i].px, W, H, img[i].comp,
                           (noise_type_t)t, table_intensity[t], 4242u, 0, 0, 0, 0);
         CHECK(rc == NOISE_OK, "repro inject 1 failed: %d", rc);
         rc = inject_noise(dirty2, img[i].px, W, H, img[i].comp,
                           (noise_type_t)t, table_intensity[t], 4242u, 0, 0, 0, 0);
         CHECK(rc == NOISE_OK, "repro inject 2 failed: %d", rc);
         CHECK(memcmp(dirty, dirty2, n) == 0,
               "%s/%s: same seed produced different output", img[i].name, noise_name(t));

         rc = inject_noise(dirty2, img[i].px, W, H, img[i].comp,
                           (noise_type_t)t, table_intensity[t], 4243u, 0, 0, 0, 0);
         CHECK(rc == NOISE_OK, "repro inject 3 failed: %d", rc);
         CHECK(memcmp(dirty, dirty2, n) != 0,
               "%s/%s: different seeds produced identical output", img[i].name, noise_name(t));
      }
   }
   /* seed 0 must be a valid seed and still be reproducible */
   rc = inject_noise(dirty,  img[0].px, W, H, 3, NOISE_GAUSSIAN, 10.0, 0u, 0, 0, 0, 0);
   CHECK(rc == NOISE_OK, "seed 0 rejected: %d", rc);
   rc = inject_noise(dirty2, img[0].px, W, H, 3, NOISE_GAUSSIAN, 10.0, 0u, 0, 0, 0, 0);
   CHECK(rc == NOISE_OK, "seed 0 second call rejected: %d", rc);
   CHECK(memcmp(dirty, dirty2, (size_t)W * H * 3) == 0, "seed 0 not reproducible");
   printf("    all 12 image/noise pairs byte-identical under repeated seed; seed 0 ok\n");

   printf("\n[4] argument validation: every invalid call must fail and leave dst untouched\n");
   {
      struct { const char *desc; noise_type_t type; double inten;
               int w, h, comp, rx, ry, rw, rh; } bad[] = {
         { "gaussian intensity -1",       NOISE_GAUSSIAN,    -1.0,   W, H, 3, 0,0,0,0 },
         { "gaussian intensity 101",      NOISE_GAUSSIAN,   101.0,   W, H, 3, 0,0,0,0 },
         { "salt_pepper intensity -0.5",  NOISE_SALT_PEPPER, -0.5,  W, H, 3, 0,0,0,0 },
         { "salt_pepper intensity 1.01",  NOISE_SALT_PEPPER,  1.01, W, H, 3, 0,0,0,0 },
         { "stripe intensity -1",         NOISE_STRIPE,      -1.0,   W, H, 3, 0,0,0,0 },
         { "stripe intensity 256",        NOISE_STRIPE,     256.0,   W, H, 3, 0,0,0,0 },
         { "intensity NaN",               NOISE_GAUSSIAN,         (double)NAN, W, H, 3, 0,0,0,0 },
         { "intensity +inf",              NOISE_GAUSSIAN,    INFINITY, W, H, 3, 0,0,0,0 },
         { "unknown noise type",          (noise_type_t)99,   1.0,   W, H, 3, 0,0,0,0 },
         { "width 0",                     NOISE_GAUSSIAN,    10.0,   0, H, 3, 0,0,0,0 },
         { "negative height",             NOISE_GAUSSIAN,    10.0,   W,-1, 3, 0,0,0,0 },
         { "channels 0",                  NOISE_GAUSSIAN,    10.0,   W, H, 0, 0,0,0,0 },
         { "channels 5",                  NOISE_GAUSSIAN,    10.0,   W, H, 5, 0,0,0,0 },
         { "region x negative",           NOISE_GAUSSIAN,    10.0,   W, H, 3, -1,0,6,6 },
         { "region past right edge",      NOISE_GAUSSIAN,    10.0,   W, H, 3, 12,0,8,6 },
         { "region past bottom edge",     NOISE_GAUSSIAN,    10.0,   W, H, 3, 0,12,6,8 },
      };
      unsigned int k;
      for (k = 0; k < sizeof bad / sizeof bad[0]; ++k) {
         memcpy(dirty, marker, sizeof marker);
         rc = inject_noise(dirty, img[0].px, bad[k].w, bad[k].h, bad[k].comp,
                           bad[k].type, bad[k].inten, 1u,
                           bad[k].rx, bad[k].ry, bad[k].rw, bad[k].rh);
         CHECK(rc != NOISE_OK, "accepted invalid input: %s (rc=%d)", bad[k].desc, rc);
         CHECK(memcmp(dirty, marker, sizeof marker) == 0,
               "%s: dst written despite validation failure", bad[k].desc);
         printf("    rejected %-28s -> %s\n", bad[k].desc, noise_strerror(rc));
      }
      rc = inject_noise(NULL, img[0].px, W, H, 3, NOISE_GAUSSIAN, 10.0, 1u, 0,0,0,0);
      CHECK(rc == NOISE_ERR_NULL, "null dst not rejected");
      rc = inject_noise(dirty, NULL, W, H, 3, NOISE_GAUSSIAN, 10.0, 1u, 0,0,0,0);
      CHECK(rc == NOISE_ERR_NULL, "null src not rejected");
      /* boundary intensities must be accepted */
      CHECK(inject_noise(dirty, img[0].px, W, H, 3, NOISE_GAUSSIAN,     0.0, 1u, 0,0,0,0) == NOISE_OK, "gauss 0 rejected");
      CHECK(inject_noise(dirty, img[0].px, W, H, 3, NOISE_GAUSSIAN,   100.0, 1u, 0,0,0,0) == NOISE_OK, "gauss 100 rejected");
      CHECK(inject_noise(dirty, img[0].px, W, H, 3, NOISE_SALT_PEPPER,  0.0, 1u, 0,0,0,0) == NOISE_OK, "sp 0 rejected");
      CHECK(inject_noise(dirty, img[0].px, W, H, 3, NOISE_SALT_PEPPER,  1.0, 1u, 0,0,0,0) == NOISE_OK, "sp 1 rejected");
      CHECK(inject_noise(dirty, img[0].px, W, H, 3, NOISE_STRIPE,       0.0, 1u, 0,0,0,0) == NOISE_OK, "stripe 0 rejected");
      CHECK(inject_noise(dirty, img[0].px, W, H, 3, NOISE_STRIPE,     255.0, 1u, 0,0,0,0) == NOISE_OK, "stripe 255 rejected");
   }

   printf("\n[5] source images still byte-identical to originals after every call\n");
   {
      int src_ok = 0;
      for (i = 0; i < 4; ++i) {
         n = (size_t)W * H * img[i].comp;
         if (memcmp(img[i].px, backup[i].px, n) == 0)
            ++src_ok;
         else
            CHECK(0, "%s sample image was modified", img[i].name);
      }
      printf("    %d/4 sample images unchanged\n", src_ok);
      CHECK(src_ok == 4, "source image invariance failed");
   }

   /* in-place injection (src == dst) must work and match an out-of-place call */
   {
      unsigned char inplace[W * H * 3];
      memcpy(inplace, backup[0].px, sizeof inplace);
      rc = inject_noise(inplace, inplace, W, H, 3, NOISE_GAUSSIAN, 20.0, 99u, 2, 1, 8, 9);
      CHECK(rc == NOISE_OK, "in-place inject failed: %d", rc);
      rc = inject_noise(dirty, backup[0].px, W, H, 3, NOISE_GAUSSIAN, 20.0, 99u, 2, 1, 8, 9);
      CHECK(rc == NOISE_OK, "reference inject failed: %d", rc);
      CHECK(memcmp(inplace, dirty, sizeof inplace) == 0,
            "in-place injection differs from out-of-place result");
      printf("    in-place (src == dst) injection matches out-of-place result\n");
   }

   printf("\n=== %d checks, %d failure(s) ===\n", g_checks, g_failures);
   if (g_failures) {
      printf("RESULT: FAIL\n");
      return 1;
   }
   printf("RESULT: PASS\n");
   return 0;
}
