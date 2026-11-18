// tests/poisson_blend.c
//
// Gradient-domain (simplified Poisson) blending + acceptance tests.
//
// Takes the source image's gradient field inside a masked region and solves
// the Poisson equation (Gauss-Seidel with SOR, configurable iteration count
// and tolerance) so the pasted region is seamless. The region boundary is a
// Dirichlet condition, optionally taken from the target image's colors.
//
// Acceptance cases (all 512x512, generated procedurally, no external data):
//   1. large brightness difference
//   2. large color difference
//   3. thin elongated region
//   4. source with alpha channel
//
// Reports per case: seam gradient jump (vs direct composite), difference to
// direct composite, interior texture preservation (gradient correlation with
// the source), iterations used, elapsed time. Also verifies that max_iter=0
// degrades to a direct composite.
//
// Build (from repo root):  cc -O2 -o poisson_blend tests/poisson_blend.c -lm
// Run   (from repo root):  ./poisson_blend        (writes artifacts/poisson_*.png)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

#define W 512
#define H 512

/* ------------------------------------------------------------------ */
/* Poisson blending                                                    */
/* ------------------------------------------------------------------ */

typedef struct {
   int max_iter;              /* 0 => degrade to direct composite */
   float tol;                 /* stop when max update per pixel < tol (0..255 scale) */
   float omega;               /* SOR relaxation; 1.0 = plain Gauss-Seidel */
   int boundary_from_target;  /* 1: Dirichlet boundary = target image colors;
                                 0: boundary = direct-composite colors */
   int iterations_used;       /* out: max per-channel iterations actually run */
} poisson_opts;

/* direct overlay: alpha-over for RGBA, masked copy for RGB */
static void composite(unsigned char *out, const unsigned char *tgt, const unsigned char *src,
                      const unsigned char *mask, int w, int h, int c)
{
   int n = w*h, i, ch;
   if (c == 4) {
      for (i = 0; i < n; ++i) {
         float a = src[i*4+3] / 255.0f;
         for (ch = 0; ch < 3; ++ch)
            out[i*4+ch] = (unsigned char)(src[i*4+ch]*a + tgt[i*4+ch]*(1.0f-a) + 0.5f);
         out[i*4+3] = 255;
      }
   } else {
      for (i = 0; i < n; ++i) {
         int m = mask ? mask[i] : 1;
         const unsigned char *s = m ? src+i*c : tgt+i*c;
         for (ch = 0; ch < c; ++ch) out[i*c+ch] = s[ch];
      }
   }
}

/* one Gauss-Seidel/SOR solve for a single channel.
   f: current solution (in/out), g: guidance (source), bnd: Dirichlet values
   for pixels outside the mask, list: masked pixel indices. */
static int solve_channel(float *f, const float *g, const float *bnd,
                         const unsigned char *mask, int w, int h,
                         const int *list, int m, int max_iter, float tol, float omega)
{
   int it, i;
   for (it = 0; it < max_iter; ++it) {
      float maxd = 0.0f;
      for (i = 0; i < m; ++i) {
         int p = list[i], x = p % w;
         float sp = g[p], sum = 0.0f, nv, d;
         int cnt = 0, q;
         if (x > 0)     { q = p-1; sum += mask[q] ? f[q]+sp-g[q] : bnd[q]; ++cnt; }
         if (x < w-1)   { q = p+1; sum += mask[q] ? f[q]+sp-g[q] : bnd[q]; ++cnt; }
         if (p >= w)    { q = p-w; sum += mask[q] ? f[q]+sp-g[q] : bnd[q]; ++cnt; }
         if (p < w*(h-1)){ q = p+w; sum += mask[q] ? f[q]+sp-g[q] : bnd[q]; ++cnt; }
         nv = sum / cnt;
         d = (nv - f[p]) * omega;
         f[p] += d;
         d = fabsf(d);
         if (d > maxd) maxd = d;
      }
      if (maxd < tol) { ++it; break; }
   }
   return it;
}

/* Blend src into tgt over the masked region with gradient-domain fusion.
   out/tgt/src are w*h*c (c = 3 or 4). For c=4 the mask is derived from the
   source alpha (>=128) and `mask` may be NULL; for c=3 `mask` (w*h bytes)
   selects the pasted region. Result written to `out`. */
static void poisson_blend(unsigned char *out, const unsigned char *tgt, const unsigned char *src,
                          const unsigned char *mask, int w, int h, int c, poisson_opts *o)
{
   int n = w*h, nc = c > 3 ? 3 : c, ch, i, m = 0;
   unsigned char *amask = NULL;
   const unsigned char *imask = mask;
   int *list;
   float *f, *g, *b;
   const unsigned char *bnd8;

   composite(out, tgt, src, mask, w, h, c);
   o->iterations_used = 0;
   if (o->max_iter <= 0) return;      /* degrade to direct composite */

   if (c == 4) {
      amask = (unsigned char *)malloc(n);
      for (i = 0; i < n; ++i) amask[i] = src[i*4+3] >= 128;
      imask = amask;
   }
   if (!imask) { free(amask); return; }

   list = (int *)malloc(n*sizeof(int));
   for (i = 0; i < n; ++i) if (imask[i]) list[m++] = i;
   if (!m) { free(list); free(amask); return; }

   bnd8 = o->boundary_from_target ? tgt : out;

   f = (float *)malloc(n*sizeof(float));
   g = (float *)malloc(n*sizeof(float));
   b = (float *)malloc(n*sizeof(float));

   for (ch = 0; ch < nc; ++ch) {
      double acc = 0.0;
      int cnt = 0, it;
      float off;
      for (i = 0; i < n; ++i) {
         g[i] = (float)src[i*c+ch];
         b[i] = (float)bnd8[i*c+ch];
         f[i] = (float)out[i*c+ch];
      }
      /* initial guess: source shifted by the mean boundary mismatch, so the
         solver only has to smooth out the residual (huge speedup) */
      for (i = 0; i < m; ++i) {
         int p = list[i], x = p % w;
         if (x > 0      && !imask[p-1]) { acc += b[p-1]-g[p]; ++cnt; }
         if (x < w-1    && !imask[p+1]) { acc += b[p+1]-g[p]; ++cnt; }
         if (p >= w     && !imask[p-w]) { acc += b[p-w]-g[p]; ++cnt; }
         if (p < w*(h-1) && !imask[p+w]) { acc += b[p+w]-g[p]; ++cnt; }
      }
      off = cnt ? (float)(acc/cnt) : 0.0f;
      for (i = 0; i < m; ++i) f[list[i]] = g[list[i]] + off;

      it = solve_channel(f, g, b, imask, w, h, list, m, o->max_iter, o->tol, o->omega);
      if (it > o->iterations_used) o->iterations_used = it;

      for (i = 0; i < m; ++i) {
         float v = f[list[i]];
         v = v < 0.0f ? 0.0f : v > 255.0f ? 255.0f : v;
         out[list[i]*c+ch] = (unsigned char)(v + 0.5f);
      }
   }

   free(f); free(g); free(b); free(list); free(amask);
}

/* ------------------------------------------------------------------ */
/* metrics                                                             */
/* ------------------------------------------------------------------ */

/* mean/max absolute value jump across the mask boundary (RGB channels) */
static void seam_jump(const unsigned char *img, const unsigned char *mask,
                      int w, int h, int c, double *mean, double *mx)
{
   int nc = c > 3 ? 3 : c, x, y, ch;
   double sum = 0.0, mxd = 0.0;
   long cnt = 0;
   for (y = 0; y < h; ++y) for (x = 0; x < w; ++x) {
      int p = y*w+x, q, d2;
      for (d2 = 0; d2 < 2; ++d2) {
         if (d2 == 0) { if (x+1 >= w) continue; q = p+1; }
         else         { if (y+1 >= h) continue; q = p+w; }
         if (mask[p] == mask[q]) continue;
         double d = 0.0;
         for (ch = 0; ch < nc; ++ch) d += fabs((double)img[p*c+ch] - img[q*c+ch]);
         d /= nc;
         sum += d; if (d > mxd) mxd = d; ++cnt;
      }
   }
   *mean = cnt ? sum/cnt : 0.0;
   *mx = mxd;
}

/* Pearson correlation between result gradients and source gradients,
   over pairs fully inside the mask (texture preservation) */
static double grad_corr(const unsigned char *res, const unsigned char *src,
                        const unsigned char *mask, int w, int h, int c)
{
   int nc = c > 3 ? 3 : c, x, y, ch, d2;
   double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
   long cnt = 0;
   for (y = 0; y < h; ++y) for (x = 0; x < w; ++x) {
      int p = y*w+x, q;
      for (d2 = 0; d2 < 2; ++d2) {
         if (d2 == 0) { if (x+1 >= w) continue; q = p+1; }
         else         { if (y+1 >= h) continue; q = p+w; }
         if (!mask[p] || !mask[q]) continue;
         for (ch = 0; ch < nc; ++ch) {
            double a = (double)res[q*c+ch] - res[p*c+ch];
            double b = (double)src[q*c+ch] - src[p*c+ch];
            sx += a; sy += b; sxx += a*a; syy += b*b; sxy += a*b; ++cnt;
         }
      }
   }
   if (!cnt) return 1.0;
   {
      double num = sxy - sx*sy/cnt;
      double den = sqrt((sxx - sx*sx/cnt) * (syy - sy*sy/cnt));
      return den > 0 ? num/den : 1.0;
   }
}

/* mean absolute difference between two images (RGB channels) */
static double mean_abs_diff(const unsigned char *a, const unsigned char *b, int w, int h, int c)
{
   int nc = c > 3 ? 3 : c, i, ch;
   double s = 0.0;
   for (i = 0; i < w*h; ++i)
      for (ch = 0; ch < nc; ++ch)
         s += fabs((double)a[i*c+ch] - b[i*c+ch]);
   return s / (w*h*nc);
}

static double now_ms(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return ts.tv_sec*1000.0 + ts.tv_nsec/1e6;
}

/* ------------------------------------------------------------------ */
/* synthetic test images                                               */
/* ------------------------------------------------------------------ */

static float frand2(int x, int y, int seed)
{
   unsigned int hh = (unsigned int)x*374761393u + (unsigned int)y*668265263u
                   + (unsigned int)seed*974634211u;
   hh = (hh ^ (hh >> 13)) * 1274126177u;
   hh ^= hh >> 16;
   return (hh & 0xFFFF) / 65535.0f * 2.0f - 1.0f;
}

/* smooth texture + per-pixel grain, roughly [-1,1] */
static float texture2(int x, int y, int seed)
{
   return 0.55f*sinf(x*0.13f + seed*1.7f)*cosf(y*0.09f - seed*0.8f)
        + 0.25f*sinf((x+y)*0.31f + seed*2.3f)
        + 0.20f*frand2(x, y, seed);
}

static void fill_solid_tex(unsigned char *img, int w, int h, int c,
                           float base[3], float amp, float grain, int seed)
{
   int x, y;
   for (y = 0; y < h; ++y) for (x = 0; x < w; ++x) {
      int p = y*w+x, ch;
      for (ch = 0; ch < 3; ++ch) {
         float v = base[ch] + amp*texture2(x, y, seed + ch*7)
                 + grain*frand2(x, y, seed + 100 + ch);
         v = v < 0 ? 0 : v > 255 ? 255 : v;
         img[p*c+ch] = (unsigned char)(v + 0.5f);
      }
      if (c == 4) img[p*c+3] = 255;
   }
}

static float sd_round_box(float x, float y, float cx, float cy, float hx, float hy, float r)
{
   float qx = fabsf(x-cx) - hx + r, qy = fabsf(y-cy) - hy + r;
   float ax = qx > 0 ? qx : 0, ay = qy > 0 ? qy : 0;
   float in = qx > qy ? qx : qy;
   return sqrtf(ax*ax + ay*ay) + (in < 0 ? in : 0) - r;
}

/* ------------------------------------------------------------------ */
/* test driver                                                         */
/* ------------------------------------------------------------------ */

typedef void (*mask_fn)(unsigned char *mask, unsigned char *alpha, int w, int h);

static int g_pass = 1;

static void run_case(const char *name, const unsigned char *tgt, const unsigned char *src,
                     const unsigned char *mask, int w, int h, int c)
{
   unsigned char *comp = (unsigned char *)malloc(w*h*c);
   unsigned char *blended = (unsigned char *)malloc(w*h*c);
   unsigned char *degen = (unsigned char *)malloc(w*h*c);
   unsigned char *imask = (unsigned char *)mask;
   unsigned char *amask = NULL;
   poisson_opts o;
   double jm_c, jx_c, jm_p, jx_p, corr, diff, t0, t1;
   int i;

   if (c == 4) { /* mask from alpha for metrics */
      amask = (unsigned char *)malloc(w*h);
      for (i = 0; i < w*h; ++i) amask[i] = src[i*4+3] >= 128;
      imask = amask;
   }

   /* baseline: direct composite */
   memset(&o, 0, sizeof o);
   o.max_iter = 0;
   poisson_blend(comp, tgt, src, mask, w, h, c, &o);

   /* degeneracy: max_iter=0 must equal direct composite */
   o.max_iter = 0; o.tol = 0.25f; o.omega = 1.9f; o.boundary_from_target = 1;
   poisson_blend(degen, tgt, src, mask, w, h, c, &o);
   if (memcmp(degen, comp, (size_t)w*h*c) != 0) {
      printf("  [FAIL] max_iter=0 does not degrade to direct composite\n");
      g_pass = 0;
   }

   /* poisson blend */
   o.max_iter = 2000; o.tol = 0.25f; o.omega = 1.9f; o.boundary_from_target = 1;
   t0 = now_ms();
   poisson_blend(blended, tgt, src, mask, w, h, c, &o);
   t1 = now_ms();

   seam_jump(comp, imask, w, h, c, &jm_c, &jx_c);
   seam_jump(blended, imask, w, h, c, &jm_p, &jx_p);
   corr = grad_corr(blended, src, imask, w, h, c);
   diff = mean_abs_diff(blended, comp, w, h, c);

   {
      double red = jm_c > 0 ? (1.0 - jm_p/jm_c)*100.0 : 100.0;
      double ms = t1 - t0;
      int ok = red >= 70.0 && corr > 0.9 && ms <= 800.0;
      if (!ok) g_pass = 0;
      printf("%s\n", name);
      printf("  seam gradient jump : composite mean %7.2f max %7.2f | poisson mean %7.2f max %7.2f | reduced %.1f%% %s\n",
             jm_c, jx_c, jm_p, jx_p, red, red >= 70.0 ? "PASS" : "FAIL(need >=70%)");
      printf("  diff vs composite  : mean |d| = %.2f /255\n", diff);
      printf("  texture preserved  : gradient correlation = %.4f %s\n", corr, corr > 0.9 ? "PASS" : "FAIL(need >0.9)");
      printf("  solver             : %d iterations, %.1f ms %s\n", o.iterations_used, ms, ms <= 800.0 ? "PASS" : "FAIL(need <=800ms)");
   }

   {
      char path[256];
      snprintf(path, sizeof path, "artifacts/poisson_%s_tgt.png", name);
      stbi_write_png(path, w, h, c, tgt, w*c);
      snprintf(path, sizeof path, "artifacts/poisson_%s_src.png", name);
      stbi_write_png(path, w, h, c, src, w*c);
      snprintf(path, sizeof path, "artifacts/poisson_%s_composite.png", name);
      stbi_write_png(path, w, h, c, comp, w*c);
      snprintf(path, sizeof path, "artifacts/poisson_%s_poisson.png", name);
      stbi_write_png(path, w, h, c, blended, w*c);
   }

   free(comp); free(blended); free(degen); free(amask);
}

int main(void)
{
   unsigned char *tgt = (unsigned char *)malloc(W*H*4);
   unsigned char *src = (unsigned char *)malloc(W*H*4);
   unsigned char *mask = (unsigned char *)malloc(W*H);
   int x, y;

   /* case 1: large brightness difference, circular region */
   {
      float bt[3] = { 45, 45, 50 }, bs[3] = { 215, 210, 200 };
      fill_solid_tex(tgt, W, H, 3, bt, 18, 0, 1);
      fill_solid_tex(src, W, H, 3, bs, 25, 0, 2);
      for (y = 0; y < H; ++y) for (x = 0; x < W; ++x) {
         float dx = x-256.0f, dy = y-256.0f;
         mask[y*W+x] = dx*dx+dy*dy < 130.0f*130.0f;
      }
      run_case("1_brightness", tgt, src, mask, W, H, 3);
   }

   /* case 2: large color difference, rounded-rect region */
   {
      float bt[3] = { 200, 70, 60 }, bs[3] = { 60, 80, 210 };
      fill_solid_tex(tgt, W, H, 3, bt, 20, 0, 3);
      fill_solid_tex(src, W, H, 3, bs, 20, 0, 4);
      for (y = 0; y < H; ++y) for (x = 0; x < W; ++x)
         mask[y*W+x] = sd_round_box((float)x, (float)y, 256, 256, 110, 80, 30) < 0;
      run_case("2_color", tgt, src, mask, W, H, 3);
   }

   /* case 3: thin elongated diagonal strip (16px wide, ~550px long) */
   {
      float bt[3] = { 120, 120, 125 }, bs[3] = { 190, 185, 175 };
      fill_solid_tex(tgt, W, H, 3, bt, 20, 0, 5);
      fill_solid_tex(src, W, H, 3, bs, 25, 10, 6);
      for (y = 0; y < H; ++y) for (x = 0; x < W; ++x)
         mask[y*W+x] = fabsf(y - (0.6f*x + 80.0f)) < 8.0f;
      run_case("3_thin", tgt, src, mask, W, H, 3);
   }

   /* case 4: source with alpha (hard-edged cut-out blob, like a product cutout) */
   {
      float bt[3] = { 150, 140, 130 }, bs[3] = { 60, 210, 90 };
      fill_solid_tex(tgt, W, H, 4, bt, 20, 0, 7);
      fill_solid_tex(src, W, H, 4, bs, 25, 0, 8);
      for (y = 0; y < H; ++y) for (x = 0; x < W; ++x) {
         float d1 = sqrtf((x-220.0f)*(x-220.0f) + (y-230.0f)*(y-230.0f)) - 100.0f;
         float d2 = sqrtf((x-310.0f)*(x-310.0f) + (y-300.0f)*(y-300.0f)) - 70.0f;
         float d = d1 < d2 ? d1 : d2;
         src[(y*W+x)*4+3] = d < 0 ? 255 : 0;
      }
      run_case("4_alpha", tgt, src, NULL, W, H, 4);
   }

   printf("\n%s\n", g_pass ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
   free(tgt); free(src); free(mask);
   return g_pass ? 0 : 1;
}
