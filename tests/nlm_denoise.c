// nlm_denoise.c -- simplified Non-Local Means denoiser + acceptance tests.
//
// Self-contained: no third-party dependencies. Uses the repo's own
// stb_image_write.h only to dump small PNG samples for eyeballing.
// Threads come from POSIX pthreads (system library, not a third-party lib).
//
// Build (from repo root):
//   cc -O2 -I. -pthread tests/nlm_denoise.c -lm -o /tmp/nlm_denoise
// Run:
//   /tmp/nlm_denoise [outdir]        (default outdir: artifacts/nlm)
//
// Algorithm (simplified NLM, Buades et al.):
//   out(p) = sum_q  w(p,q) * in(q)  /  sum_q w(p,q)
//   q ranges over a (2*search_radius+1)^2 window centered at p;
//   w(p,q) = exp(-max(MSE(patch_p, patch_q) - 2*sigma^2, 0) / h^2)
//   with patches of (2*patch_radius+1)^2 pixels. The 2*sigma^2 term is the
//   expected patch distance of two identical patches under Gaussian noise,
//   subtracting it keeps weights from collapsing on flat areas.
//
// Degenerate case: h <= 0 forces uniform weights, so the filter becomes an
// exact box (mean) filter over the search window. With search_radius = 1
// that is exactly a 3x3 mean filter (verified below with 0 error).
//
// Parallel version: rows are split into bands, one pthread per band. Every
// thread reads only the (padded) input and writes only its own output rows,
// so results are bit-identical to the single-threaded run.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ------------------------------------------------------------------ */
/* config                                                              */

typedef struct {
   int search_radius;  /* search window = (2r+1)^2                     */
   int patch_radius;   /* patch size    = (2r+1)^2                     */
   float h;            /* decay; h <= 0 -> uniform weights (mean filt.)*/
   float sigma;        /* noise sigma, used for the 2*sigma^2 offset   */
   int nthreads;       /* 1 = single-threaded                          */
} NlmCfg;

/* ------------------------------------------------------------------ */
/* deterministic RNG (splitmix64 + Box-Muller), no external deps       */

static unsigned long long rng_state = 0x123456789abcdefULL;
static unsigned long long sm64(void) {
   unsigned long long z = (rng_state += 0x9E3779B97F4A7C15ULL);
   z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
   z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
   return z ^ (z >> 31);
}
static double urand(void) { return (sm64() >> 11) * (1.0 / 9007199254740992.0); }
static double gauss(void) {
   double u1 = urand(); if (u1 < 1e-12) u1 = 1e-12;
   return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * urand());
}

static double now_sec(void) {
   struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
   return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ------------------------------------------------------------------ */
/* NLM core                                                            */

typedef struct {
   const float *pad;   /* input padded by B = sr+pr replicated border  */
   float *out;
   int w, h, pw;       /* image w/h, padded row stride                 */
   int y0, y1;         /* output rows this job computes                */
   int sr, pr;
   float h2inv;        /* 1/h^2, or 0 for uniform weights              */
   float two_sigma2;
} NlmJob;

static void nlm_rows(NlmJob *j) {
   const int w = j->w, pw = j->pw, sr = j->sr, pr = j->pr;
   const int B = sr + pr;
   const int npix = (2*pr+1)*(2*pr+1);
   const float inv_npix = 1.0f / npix;
   const int uniform = (j->h2inv == 0.0f);
   for (int y = j->y0; y < j->y1; y++) {
      float *orow = j->out + (size_t)y * w;
      for (int x = 0; x < w; x++) {
         /* center patch top-left in padded coords */
         const float *cp = j->pad + (size_t)(y + B - pr) * pw + (x + B - pr);
         double sum = 0.0, wsum = 0.0;
         for (int oy = -sr; oy <= sr; oy++) {
            for (int ox = -sr; ox <= sr; ox++) {
               float wt;
               if (uniform) {
                  wt = 1.0f;
               } else {
                  const float *np = cp + oy * pw + ox;
                  float d = 0.0f;
                  for (int py = 0; py <= 2*pr; py++) {
                     const float *ra = cp + py * pw;
                     const float *rb = np + py * pw;
                     for (int px = 0; px <= 2*pr; px++) {
                        float t = ra[px] - rb[px];
                        d += t * t;
                     }
                  }
                  d = d * inv_npix - j->two_sigma2;
                  if (d < 0) d = 0;
                  wt = (float)exp(-(double)d * j->h2inv);
               }
               sum   += wt * j->pad[(size_t)(y + B + oy) * pw + (x + B + ox)];
               wsum  += wt;
            }
         }
         orow[x] = (float)(sum / wsum);
      }
   }
}

/* pad input with replicated border of B pixels */
static float *pad_image(const float *in, int w, int h, int B) {
   int pw = w + 2*B, ph = h + 2*B;
   float *p = (float *)malloc((size_t)pw * ph * sizeof(float));
   for (int y = 0; y < ph; y++) {
      int sy = y - B; if (sy < 0) sy = 0; if (sy >= h) sy = h-1;
      const float *srow = in + (size_t)sy * w;
      float *drow = p + (size_t)y * pw;
      for (int x = 0; x < pw; x++) {
         int sx = x - B; if (sx < 0) sx = 0; if (sx >= w) sx = w-1;
         drow[x] = srow[sx];
      }
   }
   return p;
}

void nlm_denoise(const float *in, float *out, int w, int h, NlmCfg cfg) {
   int B = cfg.search_radius + cfg.patch_radius;
   float *pad = pad_image(in, w, h, B);
   int nt = cfg.nthreads < 1 ? 1 : cfg.nthreads;
   if (nt > h) nt = h;

   NlmJob *jobs = (NlmJob *)malloc(sizeof(NlmJob) * nt);
   pthread_t *tid = (pthread_t *)malloc(sizeof(pthread_t) * nt);
   for (int i = 0; i < nt; i++) {
      jobs[i].pad = pad; jobs[i].out = out;
      jobs[i].w = w; jobs[i].h = h; jobs[i].pw = w + 2*B;
      jobs[i].y0 = h * i / nt; jobs[i].y1 = h * (i+1) / nt;
      jobs[i].sr = cfg.search_radius; jobs[i].pr = cfg.patch_radius;
      jobs[i].h2inv = (cfg.h <= 0) ? 0.0f : 1.0f / (cfg.h * cfg.h);
      jobs[i].two_sigma2 = 2.0f * cfg.sigma * cfg.sigma;
   }
   if (nt == 1) {
      nlm_rows(&jobs[0]);
   } else {
      for (int i = 0; i < nt; i++)
         pthread_create(&tid[i], NULL, (void *(*)(void *))nlm_rows, &jobs[i]);
      for (int i = 0; i < nt; i++)
         pthread_join(tid[i], NULL);
   }
   free(jobs); free(tid); free(pad);
}

/* ------------------------------------------------------------------ */
/* metrics                                                             */

static double mse_of(const float *a, const float *b, int n) {
   double s = 0;
   for (int i = 0; i < n; i++) { double d = a[i]-b[i]; s += d*d; }
   return s / n;
}
static double psnr(const float *a, const float *b, int n) {
   double m = mse_of(a, b, n);
   return m <= 0 ? 99.0 : 10.0 * log10(255.0*255.0 / m);
}

/* SSIM, 7x7 uniform window, standard constants for L=255 */
static double ssim(const float *a, const float *b, int w, int h) {
   const double C1 = 6.5025, C2 = 58.5225;
   const int R = 3;
   double acc = 0; long cnt = 0;
   for (int y = R; y < h-R; y++) for (int x = R; x < w-R; x++) {
      double ma=0, mb=0, va=0, vb=0, cov=0;
      for (int dy=-R; dy<=R; dy++) for (int dx=-R; dx<=R; dx++) {
         double pa = a[(y+dy)*w + x+dx], pb = b[(y+dy)*w + x+dx];
         ma += pa; mb += pb;
      }
      double n = 49.0; ma /= n; mb /= n;
      for (int dy=-R; dy<=R; dy++) for (int dx=-R; dx<=R; dx++) {
         double pa = a[(y+dy)*w + x+dx] - ma, pb = b[(y+dy)*w + x+dx] - mb;
         va += pa*pa; vb += pb*pb; cov += pa*pb;
      }
      va /= n; vb /= n; cov /= n;
      acc += ((2*ma*mb + C1) * (2*cov + C2)) / ((ma*ma + mb*mb + C1) * (va + vb + C2));
      cnt++;
   }
   return acc / cnt;
}

/* mean |gradient| (central differences) over pixels where the reference
   gradient exceeds `thresh` -- i.e. "how sharp are the edges that exist" */
static double edge_gradient(const float *img, const float *ref, int w, int h, float thresh) {
   double acc = 0; long cnt = 0;
   for (int y = 1; y < h-1; y++) for (int x = 1; x < w-1; x++) {
      double rg = fabs(ref[y*w+x+1] - ref[y*w+x-1]) + fabs(ref[(y+1)*w+x] - ref[(y-1)*w+x]);
      if (rg < thresh) continue;
      acc += fabs(img[y*w+x+1] - img[y*w+x-1]) + fabs(img[(y+1)*w+x] - img[(y-1)*w+x]);
      cnt++;
   }
   return cnt ? acc / cnt : 0;
}

/* separable Gaussian blur baseline (for the "NLM doesn't blur" contrast) */
static void gauss_blur(const float *in, float *out, int w, int h, float sigma) {
   int r = (int)ceil(3*sigma);
   float *k = (float *)malloc(sizeof(float)*(2*r+1));
   float *tmp = (float *)malloc(sizeof(float)*w*h);
   double ks = 0;
   for (int i = -r; i <= r; i++) { k[i+r] = (float)exp(-0.5*i*i/(sigma*sigma)); ks += k[i+r]; }
   for (int i = 0; i <= 2*r; i++) k[i] /= (float)ks;
   for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
      double s = 0;
      for (int i = -r; i <= r; i++) { int xx = x+i; if (xx<0) xx=0; if (xx>=w) xx=w-1; s += k[i+r]*in[y*w+xx]; }
      tmp[y*w+x] = (float)s;
   }
   for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
      double s = 0;
      for (int i = -r; i <= r; i++) { int yy = y+i; if (yy<0) yy=0; if (yy>=h) yy=h-1; s += k[i+r]*tmp[yy*w+x]; }
      out[y*w+x] = (float)s;
   }
   free(k); free(tmp);
}

/* ------------------------------------------------------------------ */
/* synthetic test images (float, 0..255)                               */

typedef void (*img_fn)(float *, int, int);

static void img_uniform(float *im, int w, int h) {
   for (int i = 0; i < w*h; i++) im[i] = 128.0f;
}
static void img_texture(float *im, int w, int h) {
   for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
      double v = sin(x*0.31)*cos(y*0.27) + 0.6*sin((x+y)*0.11) + 0.4*sin(x*0.043)*sin(y*0.067);
      im[y*w+x] = (float)(128.0 + 90.0 * v / 2.0);
   }
}
static void img_lines(float *im, int w, int h) {
   for (int i = 0; i < w*h; i++) im[i] = 0.0f;
}
static void img_edges(float *im, int w, int h) {
   for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
      float v = (x < w/2) ? 20.0f : 235.0f;
      if (x >= w/4 && x < w/2 && y >= h/4 && y < 3*h/4) v = 235.0f; /* bright square */
      if (x >= 3*w/4 && y >= h/4 && y < 3*h/4) v = 20.0f;          /* dark square   */
      im[y*w+x] = v;
   }
}

/* 1px thin lines: horizontal, vertical, diagonal.
   30 vs 220 (not 0/255) so the added Gaussian noise stays two-sided --
   with saturated 0/255 pixels, clamping the noisy image to [0,255] would
   rectify the noise and cap the achievable PSNR no matter the filter. */
static void img_lines2(float *im, int w, int h) {
   for (int i = 0; i < w*h; i++) im[i] = 30.0f;
   for (int x = 0; x < w; x++) { im[(h/3)*w + x] = 220; im[(2*h/3)*w + x] = 220; }
   for (int y = 0; y < h; y++) { im[y*w + w/3] = 220; im[y*w + 2*w/3] = 220; }
   for (int t = 0; t < h && t < w; t++) im[t*w + t] = 220;
}

static void add_noise(float *im, int n, float sigma) {
   for (int i = 0; i < n; i++) {
      float v = im[i] + (float)(sigma * gauss());
      im[i] = v < 0 ? 0 : (v > 255 ? 255 : v);
   }
}

static void save_png(const char *dir, const char *name, const float *im, int w, int h) {
   char path[512];
   unsigned char *u8 = (unsigned char *)malloc(w*h);
   for (int i = 0; i < w*h; i++) {
      float v = im[i]; v = v < 0 ? 0 : (v > 255 ? 255 : v);
      u8[i] = (unsigned char)(v + 0.5f);
   }
   snprintf(path, sizeof path, "%s/%s.png", dir, name);
   stbi_write_png(path, w, h, 1, u8, w);
   free(u8);
}

/* 3x3 box filter reference (replicated border) for the degeneracy test */
static void box3x3(const float *in, float *out, int w, int h) {
   for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
      double s = 0;
      for (int dy=-1; dy<=1; dy++) for (int dx=-1; dx<=1; dx++) {
         int yy = y+dy; if (yy<0) yy=0; if (yy>=h) yy=h-1;
         int xx = x+dx; if (xx<0) xx=0; if (xx>=w) xx=w-1;
         s += in[yy*w+xx];
      }
      out[y*w+x] = (float)(s / 9.0);
   }
}

/* ------------------------------------------------------------------ */

int main(int argc, char **argv) {
   const char *outdir = argc > 1 ? argv[1] : "artifacts/nlm";
   {
      char cmd[600];
      snprintf(cmd, sizeof cmd, "mkdir -p \"%s\"", outdir);
      if (system(cmd) != 0) { fprintf(stderr, "cannot create %s\n", outdir); return 1; }
   }

   const int W = 256, H = 256, N = W*H;
   const float SIGMA = 20.0f;
   /* default NLM config: 11x11 search window, 5x5 patches */
   NlmCfg cfg = { 5, 2, 0.0f, SIGMA, 1 };
   cfg.h = 0.9f * SIGMA;   /* decay tuned for sigma=20 */
   /* dev hooks: NLM_SR / NLM_PR / NLM_HMULT override search/patch/h */
   {  const char *e;
      if ((e = getenv("NLM_SR"))) cfg.search_radius = atoi(e);
      if ((e = getenv("NLM_PR"))) cfg.patch_radius = atoi(e);
      if ((e = getenv("NLM_HMULT"))) cfg.h = (float)atof(e) * SIGMA;
   }

   struct { const char *name; img_fn fn; } imgs[4] = {
      { "uniform",  img_uniform },
      { "texture",  img_texture },
      { "lines",    img_lines2  },
      { "edges",    img_edges   },
   };

   float *clean = malloc(sizeof(float)*N), *noisy = malloc(sizeof(float)*N);
   float *den = malloc(sizeof(float)*N), *blur = malloc(sizeof(float)*N);

   int all_pass = 1;
   printf("=== quality: %dx%d, Gaussian noise sigma=%.0f, NLM search=%dx%d patch=%dx%d h=%.1f ===\n",
          W, H, SIGMA, 2*cfg.search_radius+1, 2*cfg.search_radius+1,
          2*cfg.patch_radius+1, 2*cfg.patch_radius+1, cfg.h);
   printf("%-8s | %7s %7s %6s | %6s %6s | %7s | %s\n",
          "image", "psnr_in", "psnr_out", "gain", "ssim_in", "ssim_out", "ms", "gain>=6dB");

   for (int k = 0; k < 4; k++) {
      imgs[k].fn(clean, W, H);
      memcpy(noisy, clean, sizeof(float)*N);
      rng_state = 0xC0FFEE + k;          /* fixed seed per image: reproducible */
      add_noise(noisy, N, SIGMA);

      double t0 = now_sec();
      nlm_denoise(noisy, den, W, H, cfg);
      double ms = (now_sec() - t0) * 1e3;

      double pi = psnr(clean, noisy, N), po = psnr(clean, den, N);
      double si = ssim(clean, noisy, W, H), so = ssim(clean, den, W, H);
      int ok = (po - pi) >= 6.0;
      all_pass &= ok;
      printf("%-8s | %7.2f %7.2f %+6.2f | %.4f %.4f | %7.1f | %s\n",
             imgs[k].name, pi, po, po-pi, si, so, ms, ok ? "PASS" : "FAIL");

      char nm[64];
      snprintf(nm, sizeof nm, "%s_noisy", imgs[k].name); save_png(outdir, nm, noisy, W, H);
      snprintf(nm, sizeof nm, "%s_nlm",   imgs[k].name); save_png(outdir, nm, den,  W, H);
   }

   /* thin-line sharpness: edge gradient of denoised vs Gaussian-blurred,
      measured against the clean image's edge locations */
   printf("\n=== thin-line / edge sharpness (mean |grad| at clean edge pixels, clean=1.00) ===\n");
   for (int k = 2; k < 4; k++) {
      imgs[k].fn(clean, W, H);
      memcpy(noisy, clean, sizeof(float)*N);
      rng_state = 0xC0FFEE + k;
      add_noise(noisy, N, SIGMA);
      nlm_denoise(noisy, den, W, H, cfg);
      gauss_blur(noisy, blur, W, H, 1.5f);
      double gc = edge_gradient(clean, clean, W, H, 100);
      double gd = edge_gradient(den,   clean, W, H, 100);
      double gb = edge_gradient(blur,  clean, W, H, 100);
      int ok = gd/gc >= 0.70;          /* NLM keeps >=70% of edge gradient */
      all_pass &= ok;
      printf("%-8s | clean %.1f | nlm %.1f (%.2f) | gauss %.1f (%.2f) | nlm>=0.70: %s\n",
             imgs[k].name, gc, gd, gd/gc, gb, gb/gc, ok ? "PASS" : "FAIL");
      char nm[64];
      snprintf(nm, sizeof nm, "%s_gauss", imgs[k].name); save_png(outdir, nm, blur, W, H);
   }

   /* degeneracy: search_radius=1 + h<=0 (uniform weights) == 3x3 mean filter */
   printf("\n=== degeneracy: search window 3x3 (radius 1), uniform weights -> mean filter ===\n");
   {
      NlmCfg c1 = { 1, 1, 0.0f, SIGMA, 1 };   /* h<=0 => uniform weights */
      float *ref = malloc(sizeof(float)*N), *o1 = malloc(sizeof(float)*N);
      box3x3(noisy, ref, W, H);
      nlm_denoise(noisy, o1, W, H, c1);
      double md = 0;
      for (int i = 0; i < N; i++) { double d = fabs(o1[i]-ref[i]); if (d > md) md = d; }
      int ok = md < 1e-3;
      all_pass &= ok;
      printf("max |nlm(r=1,h=0) - box3x3| = %.3g -> %s\n", md, ok ? "PASS" : "FAIL");
      free(ref); free(o1);
   }

   /* performance + parallel consistency on 512x512 */
   printf("\n=== performance: 512x512 texture, sigma=20 ===\n");
   {
      const int PW = 512, PH = 512, PN = PW*PH;
      float *pc = malloc(sizeof(float)*PN), *pn = malloc(sizeof(float)*PN);
      float *o1 = malloc(sizeof(float)*PN), *o4 = malloc(sizeof(float)*PN);
      img_texture(pc, PW, PH);
      memcpy(pn, pc, sizeof(float)*PN);
      rng_state = 0xBEEF;
      add_noise(pn, PN, SIGMA);

      NlmCfg s1 = cfg; s1.nthreads = 1;
      double t0 = now_sec(); nlm_denoise(pn, o1, PW, PH, s1); double t1 = now_sec();
      double single = t1 - t0;

      long cores = sysconf(_SC_NPROCESSORS_ONLN);
      int nt = cores < 4 ? (int)cores : 4;
      NlmCfg s4 = cfg; s4.nthreads = nt;
      t0 = now_sec(); nlm_denoise(pn, o4, PW, PH, s4); t1 = now_sec();
      double par = t1 - t0;

      size_t diff = memcmp(o1, o4, sizeof(float)*PN);
      double speedup = single / par;
      printf("single-thread: %.3f s  (limit 1.5 s) -> %s\n", single, single <= 1.5 ? "PASS" : "FAIL");
      printf("%d threads:    %.3f s  speedup %.2fx (need >= 2x) -> %s\n",
             nt, par, speedup, speedup >= 2.0 ? "PASS" : "FAIL");
      printf("tiled vs single-thread output: %s -> %s\n",
             diff == 0 ? "bit-identical" : "DIFFERS", diff == 0 ? "PASS" : "FAIL");
      printf("psnr 512: in %.2f -> out %.2f (gain %+.2f dB)\n",
             psnr(pc, pn, PN), psnr(pc, o1, PN), psnr(pc, o1, PN) - psnr(pc, pn, PN));
      all_pass &= (single <= 1.5) && (speedup >= 2.0) && (diff == 0);
      save_png(outdir, "perf512_noisy", pn, PW, PH);
      save_png(outdir, "perf512_nlm", o1, PW, PH);
      free(pc); free(pn); free(o1); free(o4);
   }

   printf("\n%s\n", all_pass ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
   free(clean); free(noisy); free(den); free(blur);
   return all_pass ? 0 : 1;
}
