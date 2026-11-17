/*
 * rotate_resample_test.c
 *
 * Fused "rotate by 90/180/270 + resample" in ONE sampling pass, versus the
 * classic two-pass pipeline (rotate into an intermediate image, then
 * resample). For every output pixel the fused path computes the source
 * coordinate with the rotation folded in and samples the source image once
 * (nearest or bilinear), so no intermediate rotated image is materialized.
 *
 * Checks performed (all must pass for exit code 0):
 *   1. invalid rotation angles are rejected (anything not in {0,90,180,270})
 *   2. identity-scale fused output is bit-exact vs a plain rotate
 *   3. corner/boundary pixels are exact (clamp-to-edge boundary policy)
 *   4. fused vs two-pass on 90/180/270 x upscale/downscale: nearest is
 *      bit-exact, bilinear within 1/255 (PSNR reported for each combo)
 *   5. per-combo timing and live-buffer peak memory of both pipelines
 *
 * Self-contained: no external dependencies, small synthetic images, one run.
 * Build:  cc -O2 -Wall -Wextra rotate_resample_test.c -lm -o rotate_resample_test
 */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

enum { RR_NEAREST = 0, RR_BILINEAR = 1 };

/* ------------------------------------------------------------------ */
/* rotation plumbing (clockwise; rotated image is rw x rh)             */

static void rot_dims(int angle, int sw, int sh, int *rw, int *rh)
{
   if (angle == 90 || angle == 270) { *rw = sh; *rh = sw; }
   else                             { *rw = sw; *rh = sh; }
}

/* exact integer map: pixel (u,v) of the rotated image -> source (sx,sy) */
static void rot_map(int angle, int u, int v, int sw, int sh, int *sx, int *sy)
{
   switch (angle) {
      case 0:   *sx = u;      *sy = v;      break;
      case 90:  *sx = v;      *sy = sh-1-u; break;
      case 180: *sx = sw-1-u; *sy = sh-1-v; break;
      default:  *sx = sw-1-v; *sy = u;      break; /* 270 */
   }
}

/* ------------------------------------------------------------------ */
/* fused single-pass rotate + resample                                 */
/*                                                                     */
/* Returns 0 on success, nonzero on invalid arguments (return 2 when   */
/* the rotation angle is not a multiple of 90).                        */
/* Boundary policy: clamp-to-edge in the rotated output space --       */
/* sampling coordinates outside the rotated image are clamped to its   */
/* edge, and the bilinear footprint collapses to the edge pixel(s).    */

static int rr_fused_rotate_resample(
   const unsigned char *src, int sw, int sh,
   unsigned char *dst, int dw, int dh,
   int channels, int angle_deg, int filter)
{
   int rw, rh, x, y, c;
   double scx, scy;

   if (src == NULL || dst == NULL)              return 1;
   if (sw < 1 || sh < 1 || dw < 1 || dh < 1)    return 1;
   if (channels < 1 || channels > 4)            return 1;
   if (filter != RR_NEAREST && filter != RR_BILINEAR) return 1;
   if (angle_deg != 0 && angle_deg != 90 &&
       angle_deg != 180 && angle_deg != 270)    return 2; /* invalid angle */

   rot_dims(angle_deg, sw, sh, &rw, &rh);
   scx = (double) rw / (double) dw;
   scy = (double) rh / (double) dh;

   for (y = 0; y < dh; y++) {
      double fy = (y + 0.5) * scy - 0.5;   /* coords in the virtual rotated image */
      for (x = 0; x < dw; x++) {
         double fx = (x + 0.5) * scx - 0.5;
         unsigned char *out = dst + ((size_t) y * dw + x) * (size_t) channels;

         if (filter == RR_NEAREST) {
            int u, v, sx, sy;
            /* round half up, then clamp-to-edge (in rotated space) */
            u = (int) floor(fx + 0.5); if (u < 0) u = 0; if (u > rw-1) u = rw-1;
            v = (int) floor(fy + 0.5); if (v < 0) v = 0; if (v > rh-1) v = rh-1;
            rot_map(angle_deg, u, v, sw, sh, &sx, &sy);
            memcpy(out, src + ((size_t) sy * sw + sx) * (size_t) channels,
                   (size_t) channels);
         } else {
            double gx = fx, gy = fy, wx, wy;
            int x0, y0, x1, y1, sx, sy;
            const unsigned char *p00, *p10, *p01, *p11;
            if (gx < 0) gx = 0;
            if (gx > rw-1) gx = rw-1;  /* clamp-to-edge */
            if (gy < 0) gy = 0;
            if (gy > rh-1) gy = rh-1;
            x0 = (int) floor(gx); x1 = (x0+1 < rw) ? x0+1 : rw-1; wx = gx - x0;
            y0 = (int) floor(gy); y1 = (y0+1 < rh) ? y0+1 : rh-1; wy = gy - y0;
            /* fetch the 4 neighbors through the folded rotation */
            rot_map(angle_deg, x0, y0, sw, sh, &sx, &sy);
            p00 = src + ((size_t) sy * sw + sx) * (size_t) channels;
            rot_map(angle_deg, x1, y0, sw, sh, &sx, &sy);
            p10 = src + ((size_t) sy * sw + sx) * (size_t) channels;
            rot_map(angle_deg, x0, y1, sw, sh, &sx, &sy);
            p01 = src + ((size_t) sy * sw + sx) * (size_t) channels;
            rot_map(angle_deg, x1, y1, sw, sh, &sx, &sy);
            p11 = src + ((size_t) sy * sw + sx) * (size_t) channels;
            for (c = 0; c < channels; c++) {
               double val = (1.0-wy) * ((1.0-wx)*p00[c] + wx*p10[c])
                          +      wy  * ((1.0-wx)*p01[c] + wx*p11[c]);
               int iv = (int) floor(val + 0.5);
               out[c] = (unsigned char) (iv < 0 ? 0 : (iv > 255 ? 255 : iv));
            }
         }
      }
   }
   return 0;
}

/* ------------------------------------------------------------------ */
/* reference two-pass pipeline: exact rotate, then plain resample      */

static void rotate_exact(const unsigned char *src, int sw, int sh,
                         unsigned char *dst, int channels, int angle)
{
   int rw, rh, u, v;
   rot_dims(angle, sw, sh, &rw, &rh);
   for (v = 0; v < rh; v++)
      for (u = 0; u < rw; u++) {
         int sx, sy;
         rot_map(angle, u, v, sw, sh, &sx, &sy);
         memcpy(dst + ((size_t) v * rw + u) * (size_t) channels,
                src + ((size_t) sy * sw + sx) * (size_t) channels,
                (size_t) channels);
      }
}

/* ------------------------------------------------------------------ */
/* counting allocator: measures live-buffer peak of a pipeline         */

static size_t mem_current, mem_peak;

static void *rr_alloc(size_t n)
{
   size_t *p = (size_t *) malloc(n + sizeof(size_t));
   if (p == NULL) { fprintf(stderr, "out of memory\n"); exit(1); }
   *p = n;
   mem_current += n;
   if (mem_current > mem_peak) mem_peak = mem_current;
   return p + 1;
}

static void rr_release(void *p)
{
   size_t *q = ((size_t *) p) - 1;
   mem_current -= *q;
   free(q);
}

static unsigned char *pipeline_twopass(const unsigned char *src, int sw, int sh,
                                       int dw, int dh, int ch,
                                       int angle, int filter)
{
   int rw, rh;
   unsigned char *rot, *dst;
   rot_dims(angle, sw, sh, &rw, &rh);
   rot = (unsigned char *) rr_alloc((size_t) rw * rh * ch);   /* intermediate */
   rotate_exact(src, sw, sh, rot, ch, angle);
   dst = (unsigned char *) rr_alloc((size_t) dw * dh * ch);
   rr_fused_rotate_resample(rot, rw, rh, dst, dw, dh, ch, 0, filter);
   rr_release(rot);
   return dst;
}

static unsigned char *pipeline_fused(const unsigned char *src, int sw, int sh,
                                     int dw, int dh, int ch,
                                     int angle, int filter)
{
   unsigned char *dst = (unsigned char *) rr_alloc((size_t) dw * dh * ch);
   rr_fused_rotate_resample(src, sw, sh, dst, dw, dh, ch, angle, filter);
   return dst;
}

/* ------------------------------------------------------------------ */
/* helpers                                                             */

static double psnr(const unsigned char *a, const unsigned char *b, size_t n,
                   int *maxdiff)
{
   double mse = 0.0;
   int md = 0;
   size_t i;
   for (i = 0; i < n; i++) {
      int d = (int) a[i] - (int) b[i];
      if (d < 0) d = -d;
      if (d > md) md = d;
      mse += (double) d * (double) d;
   }
   *maxdiff = md;
   if (mse == 0.0) return INFINITY;
   mse /= (double) n;
   return 10.0 * log10(255.0 * 255.0 / mse);
}

static double now_sec(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
}

static volatile unsigned long sink; /* keeps the optimizer honest */

static double bench(unsigned char *(*pipe)(const unsigned char *, int, int,
                                           int, int, int, int, int),
                    const unsigned char *src, int sw, int sh,
                    int dw, int dh, int ch, int angle, int filter, int iters)
{
   double t0, t1;
   int k;
   /* warmup */
   for (k = 0; k < 3; k++) {
      unsigned char *d = pipe(src, sw, sh, dw, dh, ch, angle, filter);
      sink += d[0];
      rr_release(d);
   }
   t0 = now_sec();
   for (k = 0; k < iters; k++) {
      unsigned char *d = pipe(src, sw, sh, dw, dh, ch, angle, filter);
      sink += d[((unsigned) k * 997u) % ((size_t) dw * dh * ch)];
      rr_release(d);
   }
   t1 = now_sec();
   return (t1 - t0) * 1000.0 / (double) iters; /* ms per iteration */
}

/* ------------------------------------------------------------------ */

static int failures = 0;

static void check(int cond, const char *what)
{
   if (!cond) {
      printf("FAIL: %s\n", what);
      failures++;
   }
}

int main(void)
{
   enum { SW = 96, SH = 72, CH = 3 };
   enum { UW = 200, UH = 150, DW = 40, DH = 30 };
   static unsigned char img[SW * SH * CH];
   static const int angles[3] = { 90, 180, 270 };
   static const char *fname[2] = { "nearest ", "bilinear" };
   int x, y, i, a, f;

   /* synthetic source image: smooth gradients + high-frequency pattern */
   for (y = 0; y < SH; y++)
      for (x = 0; x < SW; x++) {
         unsigned char *p = img + ((size_t) y * SW + x) * CH;
         p[0] = (unsigned char) (x * 255 / (SW - 1));
         p[1] = (unsigned char) (y * 255 / (SH - 1));
         p[2] = (unsigned char) (((x * x + y * y) >> 2) ^ ((x & 4) ? 0xFF : 0x00));
      }

   /* ---- test 1: invalid rotation angles must be rejected ----------- */
   {
      static const int bad[] = { 1, 45, 89, 91, 135, 179, 360, 720, -90, -270 };
      static const int good[] = { 0, 90, 180, 270 };
      unsigned char tmp[4];
      for (i = 0; i < (int) (sizeof(bad) / sizeof(bad[0])); i++)
         check(rr_fused_rotate_resample(img, SW, SH, tmp, 1, 1, CH,
                                        bad[i], RR_NEAREST) != 0,
               "invalid rotation angle must return an error");
      for (i = 0; i < (int) (sizeof(good) / sizeof(good[0])); i++)
         check(rr_fused_rotate_resample(img, SW, SH, tmp, 1, 1, CH,
                                        good[i], RR_NEAREST) == 0,
               "multiple-of-90 rotation angle must be accepted");
      check(rr_fused_rotate_resample(NULL, SW, SH, tmp, 1, 1, CH, 90,
                                     RR_NEAREST) != 0,
            "NULL source must return an error");
      check(rr_fused_rotate_resample(img, SW, SH, tmp, 0, 1, CH, 90,
                                     RR_NEAREST) != 0,
            "zero output width must return an error");
      printf("[1] invalid-angle / bad-argument rejection: %s\n",
             failures ? "FAILED" : "ok");
   }

   /* ---- test 2: identity-scale fused == plain rotate, bit-exact ----- */
   for (a = 0; a < 4; a++) {
      int angle = a * 90, rw, rh;
      static unsigned char ref[SW * SH * CH], out[SW * SH * CH];
      rot_dims(angle, SW, SH, &rw, &rh);
      rotate_exact(img, SW, SH, ref, CH, angle);
      for (f = 0; f < 2; f++) {
         int rc = rr_fused_rotate_resample(img, SW, SH, out, rw, rh, CH,
                                           angle, f);
         check(rc == 0 && memcmp(ref, out, (size_t) rw * rh * CH) == 0,
               "identity-scale fused output must equal plain rotate exactly");
      }
   }
   printf("[2] identity-scale rotation bit-exact (0/90/180/270, both filters): %s\n",
          failures ? "FAILED" : "ok");

   /* ---- test 3: boundary pixels (clamp-to-edge) --------------------- */
   /* 5x4 pattern with distinct pixels.
      upscale: output corners sample from outside the rotated image, so
               clamp-to-edge must reproduce the rotated corners exactly;
      downscale: border pixels are interior samples -- fused must match
               the two-pass reference bit-for-bit along the whole frame. */
   {
      enum { PW = 5, PH = 4 };
      unsigned char pat[PW * PH * CH];
      static const int up[2] = { 13, 11 }, dn[2] = { 3, 2 };
      int corner_fail = 0, border_fail = 0;
      for (y = 0; y < PH; y++)
         for (x = 0; x < PW; x++) {
            unsigned char *p = pat + ((size_t) y * PW + x) * CH;
            p[0] = (unsigned char) (x * 40 + y * 7 + 1);
            p[1] = (unsigned char) (y * 60 + x * 3 + 2);
            p[2] = (unsigned char) (x * 11 + y * 23 + 3);
         }
      for (a = 0; a < 3; a++) {
         int rw, rh, f, k;
         unsigned char rot[PH * PW * CH]; /* rotated: at most 5x4 */
         unsigned char out[13 * 11 * CH], ref[3 * 2 * CH];
         static const int cx[4] = { 0, 12, 0, 12 };  /* corners of 13x11 */
         static const int cy[4] = { 0, 0, 10, 10 };
         rot_dims(angles[a], PW, PH, &rw, &rh);
         rotate_exact(pat, PW, PH, rot, CH, angles[a]);
         for (f = 0; f < 2; f++) {
            /* upscale: corners clamp to the rotated edge pixels */
            rr_fused_rotate_resample(pat, PW, PH, out, up[0], up[1], CH,
                                     angles[a], f);
            for (k = 0; k < 4; k++) {
               int rx = (k & 1) ? rw - 1 : 0;
               int ry = (k & 2) ? rh - 1 : 0;
               if (memcmp(out + ((size_t) cy[k] * up[0] + cx[k]) * CH,
                          rot + ((size_t) ry * rw + rx) * CH, CH) != 0)
                  corner_fail = 1;
            }
            /* downscale: fused == two-pass reference, whole frame */
            rr_fused_rotate_resample(pat, PW, PH, out, dn[0], dn[1], CH,
                                     angles[a], f);
            rotate_exact(pat, PW, PH, rot, CH, angles[a]);
            rr_fused_rotate_resample(rot, rw, rh, ref, dn[0], dn[1], CH, 0, f);
            if (memcmp(out, ref, (size_t) dn[0] * dn[1] * CH) != 0)
               border_fail = 1;
         }
      }
      check(!corner_fail, "upscale corners must clamp to rotated edge pixels");
      check(!border_fail, "downscale frame must match two-pass reference");
      printf("[3] boundary pixels: clamp-to-edge, upscale corners exact + "
             "downscale frame bit-exact vs reference (90/180/270, both "
             "filters): %s\n", (corner_fail || border_fail) ? "FAILED" : "ok");
   }

   /* ---- tests 4+5: fused vs two-pass, full matrix ------------------- */
   printf("\n[4] fused (1-pass) vs rotate+resample (2-pass), source %dx%d RGB\n",
          SW, SH);
   printf("angle scale   filter    output   PSNR(dB) maxdiff |"
          " 2-pass ms  fused ms speedup | 2-pass mem  fused mem  saved\n");
   for (a = 0; a < 3; a++) {
      for (i = 0; i < 2; i++) { /* 0 = upscale, 1 = downscale */
         int dw = i ? DW : UW, dh = i ? DH : UH;
         int iters = i ? 1000 : 200;
         size_t n = (size_t) dw * dh * CH;
         for (f = 0; f < 2; f++) {
            unsigned char *ref, *out;
            size_t peak2, peak1;
            double q, t2, t1;
            int md;

            mem_current = mem_peak = 0;
            ref = pipeline_twopass(img, SW, SH, dw, dh, CH, angles[a], f);
            peak2 = mem_peak;
            mem_current = mem_peak = 0;
            out = pipeline_fused(img, SW, SH, dw, dh, CH, angles[a], f);
            peak1 = mem_peak;

            q = psnr(ref, out, n, &md);
            if (f == RR_NEAREST)
               check(md == 0, "nearest: fused must be bit-exact vs two-pass");
            else
               check(md <= 1, "bilinear: fused must be within 1/255 of two-pass");

            t2 = bench(pipeline_twopass, img, SW, SH, dw, dh, CH, angles[a], f, iters);
            t1 = bench(pipeline_fused,  img, SW, SH, dw, dh, CH, angles[a], f, iters);

            {
               char qbuf[16];
               if (isinf(q)) snprintf(qbuf, sizeof(qbuf), "inf");
               else          snprintf(qbuf, sizeof(qbuf), "%.2f", q);
               printf(" %3d  %-7s %s %3dx%-3d %8s %6d |"
                      " %9.4f %8.4f %6.2fx | %9zu  %9zu %7zu\n",
                      angles[a], i ? "down" : "up", fname[f], dw, dh,
                      qbuf, md,
                      t2, t1, t2 / t1, peak2, peak1, peak2 - peak1);
            }

            rr_release(ref);
            rr_release(out);
         }
      }
   }

   printf("\n%s (%d failure%s)  [sink=%lu]\n",
          failures ? "TESTS FAILED" : "ALL TESTS PASSED",
          failures, failures == 1 ? "" : "s", sink);
   return failures ? 1 : 0;
}
