// stb_multiexpose - v0.1 - public domain
//
// Multi-exposure blending of a bracketed burst of phone photos.
//
// The pipeline has three explicit stages, each exposed separately:
//
//   1. stbme_align()          - brightness-robust block matching with
//                               integer search plus sub-pixel parabolic
//                               refinement. On failure it falls back to
//                               zero shifts and reports the reason in the
//                               returned plan; it never returns a guessed
//                               alignment silently.
//
//   2. stbme_compute_weights() - per-pixel exposure-quality weights
//                               (Gaussian well-centred on mid-tone) with
//                               a ghost-rejection factor for frames that
//                               disagree with the reference after warping.
//
//   3. stbme_fuse()            - weighted bilinear fusion followed by a
//                               parameterised S-curve tone map.
//
// stbme_blend() runs all three stages in one call.
//
// Compiling:
//
//   In one C/C++ file that #includes this file, do
//
//      #define STB_MULTIEXPOSE_IMPLEMENTATION
//      #include "stb_multiexpose.h"
//
//   Optionally #define STB_MULTIEXPOSE_STATIC before including to make the
//   definitions private to that translation unit.
//
// License: see end of file.
//
// Supported input: 8-bit interleaved images, 1/2/3/4 components
// (alpha is ignored when present). All frames must share dimensions and
// component count. Input buffers are read-only and never modified.
// No allocations are performed after alignment returns (the plan owns
// everything it allocates; release it with stbme_free_plan()).

#ifndef INCLUDE_STB_MULTIEXPOSE_H
#define INCLUDE_STB_MULTIEXPOSE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef STBMEDEF
 #ifdef STB_MULTIEXPOSE_STATIC
  #define STBMEDEF static
 #else
  #define STBMEDEF extern
 #endif
#endif

#define STBME_MAX_FRAMES 16
#define STBME_MAX_SHIFT  32

typedef struct stbme_config
{
   int   search_range;    // integer-pixel search radius, >=1 (clamped to STBME_MAX_SHIFT)
   int   block_size;      // square block side, >=4 (clamped to [4,64])
   float weight_gamma;    // >0; smaller => narrower mid-tone well => sharper weight curve
   float tone_strength;   // [0,1]; 0 = identity, 1 = strongest S-curve
   float fallback_sad;    // >0; normalised zero-mean SAD above which a block is untrusted
   float trust_fraction;  // [0,1]; minimum fraction of blocks that must be trusted
} stbme_config;

// fallback_reason values stored in the plan
#define STBME_FB_NONE            0  // alignment succeeded
#define STBME_FB_BAD_INPUT       1  // invalid arguments (plan allocation also returns NULL)
#define STBME_FB_UNTRUSTED       2  // too few trusted blocks / shifts pinned to search edge

typedef struct stbme_plan
{
   int w, h, comp, n;
   stbme_config cfg;
   float shift_x[STBME_MAX_FRAMES]; // warp offset applied to each frame; frame 0 is always (0,0)
   float shift_y[STBME_MAX_FRAMES];
   int   aligned;                   // 1 = real alignment, 0 = explicit zero-shift fallback
   int   fallback_reason;           // one of STBME_FB_*
   float mean_sad;                  // mean best normalised SAD over trusted blocks (0 if none)
   float align_error;               // mean px distance of trusted blocks from global shift (-1 on fallback)
   int   trusted_blocks;
   int   total_blocks;
} stbme_plan;

STBMEDEF void stbme_config_default(stbme_config *cfg);

// Stage 1: align every frame against frame 0. Never returns NULL for valid
// arguments; inspect plan->aligned / fallback_reason for the fallback path.
// Returns NULL only for invalid arguments or allocation failure.
STBMEDEF stbme_plan *stbme_align(const unsigned char * const *frames, int n,
                                 int w, int h, int comp, const stbme_config *cfg);

// Stage 2: per-pixel weights, n * w * h floats, normalised so the weights
// of the n frames sum to 1 at every pixel. Uses the shifts from the plan
// (zero shifts when alignment fell back).
STBMEDEF void stbme_compute_weights(const stbme_plan *plan,
                                    const unsigned char * const *frames,
                                    float *weights);

// Stage 3: weighted bilinear fusion of the frames plus S-curve tone mapping.
// out must hold w * h * comp unsigned chars.
STBMEDEF void stbme_fuse(const stbme_plan *plan,
                         const unsigned char * const *frames,
                         const float *weights, unsigned char *out);

// All-in-one convenience wrapper. Returns the same plan stbme_align would
// return (including its fallback flags), or NULL on invalid arguments.
STBMEDEF stbme_plan *stbme_blend(const unsigned char * const *frames, int n,
                                 int w, int h, int comp, const stbme_config *cfg,
                                 unsigned char *out);

STBMEDEF void stbme_free_plan(stbme_plan *plan);

#ifdef __cplusplus
}
#endif

// ============================================================================
//  Implementation
// ============================================================================
#ifdef STB_MULTIEXPOSE_IMPLEMENTATION

#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef STBME_MAX
#define STBME_MAX(a,b) ((a) > (b) ? (a) : (b))
#endif
#ifndef STBME_MIN
#define STBME_MIN(a,b) ((a) < (b) ? (a) : (b))
#endif
#ifndef STBME_CLAMP
#define STBME_CLAMP(x,lo,hi) STBME_MIN(STBME_MAX((x),(lo)),(hi))
#endif

STBMEDEF void stbme_config_default(stbme_config *cfg)
{
   if (!cfg) return;
   cfg->search_range   = 4;
   cfg->block_size     = 8;
   cfg->weight_gamma   = 1.0f;
   cfg->tone_strength  = 0.5f;
   cfg->fallback_sad   = 0.10f;
   cfg->trust_fraction = 0.50f;
}

static float stbme_luma(const unsigned char *p, int comp)
{
   if (comp <= 2) return (float) p[0];
   return (77.0f * p[0] + 150.0f * p[1] + 29.0f * p[2]) / 256.0f;
}

static float *stbme_make_luma(const unsigned char * const *frames, int n,
                              int w, int h, int comp)
{
   int i, count = n * w * h;
   float *lum = (float *) malloc((size_t) count * sizeof(float));
   if (!lum) return NULL;
   for (i = 0; i < n; ++i) {
      const unsigned char *f = frames[i];
      float *dst = lum + (size_t) i * w * h;
      int p;
      for (p = 0; p < w * h; ++p)
         dst[p] = stbme_luma(f + (size_t) p * comp, comp);
   }
   return lum;
}

// Bilinear sample with edge clamp; coordinates are in image space.
static float stbme_samplef(const float *img, int w, int h, float x, float y)
{
   int x0, y0, x1, y1;
   float fx, fy;
   if (x <= 0) x = 0; else if (x >= (float) (w - 1)) x = (float) (w - 1);
   if (y <= 0) y = 0; else if (y >= (float) (h - 1)) y = (float) (h - 1);
   x0 = (int) x; y0 = (int) y;
   x1 = STBME_MIN(x0 + 1, w - 1);
   y1 = STBME_MIN(y0 + 1, h - 1);
   fx = x - (float) x0;
   fy = y - (float) y0;
   return (img[y0 * w + x0] * (1.0f - fx) + img[y0 * w + x1] * fx) * (1.0f - fy)
        + (img[y1 * w + x0] * (1.0f - fx) + img[y1 * w + x1] * fx) * fy;
}

static unsigned char stbme_samplec(const unsigned char *img, int comp,
                                   int w, int h, float x, float y, int c)
{
   int x0, y0, x1, y1;
   float fx, fy, v;
   if (x <= 0) x = 0; else if (x >= (float) (w - 1)) x = (float) (w - 1);
   if (y <= 0) y = 0; else if (y >= (float) (h - 1)) y = (float) (h - 1);
   x0 = (int) x; y0 = (int) y;
   x1 = STBME_MIN(x0 + 1, w - 1);
   y1 = STBME_MIN(y0 + 1, h - 1);
   fx = x - (float) x0;
   fy = y - (float) y0;
   v = (img[(size_t) (y0 * w + x0) * comp + c] * (1.0f - fx)
      + img[(size_t) (y0 * w + x1) * comp + c] * fx) * (1.0f - fy)
     + (img[(size_t) (y1 * w + x0) * comp + c] * (1.0f - fx)
      + img[(size_t) (y1 * w + x1) * comp + c] * fx) * fy;
   return (unsigned char) STBME_CLAMP(v + 0.5f, 0.0f, 255.0f);
}

// Zero-mean normalised SAD between reference block B (at bx,by) and a
// candidate block of frame k offset by (dx,dy). Brightness differences
// between exposures are removed by subtracting each block's own mean.
static float stbme_block_sad(const float *ref, const float *frm,
                             int w, int h, int bs,
                             int bx, int by, int dx, int dy)
{
   int ax = bx + dx, ay = by + dy;
   int x, y;
   double sum_b = 0, sum_a = 0, sad = 0;
   int npx = bs * bs;
   if (ax < 0 || ay < 0 || ax + bs > w || ay + bs > h)
      return 1.0f; // candidate outside frame: worst possible score
   for (y = 0; y < bs; ++y) {
      for (x = 0; x < bs; ++x) {
         sum_b += ref[(by + y) * w + (bx + x)];
         sum_a += frm[(ay + y) * w + (ax + x)];
      }
   }
   {
      float mb = (float) (sum_b / npx);
      float ma = (float) (sum_a / npx);
      for (y = 0; y < bs; ++y) {
         for (x = 0; x < bs; ++x) {
            float db = ref[(by + y) * w + (bx + x)] - mb;
            float da = frm[(ay + y) * w + (ax + x)] - ma;
            float d = da - db;
            sad += (double) (d < 0 ? -d : d);
         }
      }
   }
   return (float) (sad / ((double) npx * 255.0));
}

// Vertex offset of the parabola through (-1,sm),(0,s0),(1,sp); 0 if flat.
static float stbme_parabola(float sm, float s0, float sp)
{
   float denom = sm - 2.0f * s0 + sp;
   float d;
   if (denom <= 0.0f) return 0.0f;
   d = 0.5f * (sm - sp) / denom;
   return STBME_CLAMP(d, -0.75f, 0.75f);
}

static void stbme_insertion_sort(float *a, int n)
{
   int i, j;
   for (i = 1; i < n; ++i) {
      float v = a[i];
      for (j = i; j > 0 && a[j - 1] > v; --j)
         a[j] = a[j - 1];
      a[j] = v;
   }
}

static int stbme_validate(int n, int w, int h, int comp)
{
   if (w <= 0 || h <= 0) return 0;
   if (comp != 1 && comp != 2 && comp != 3 && comp != 4) return 0;
   if (n <= 0 || n > STBME_MAX_FRAMES) return 0;
   return 1;
}

static void stbme_apply_fallback(stbme_plan *plan, int reason)
{
   int k;
   for (k = 0; k < plan->n; ++k) {
      plan->shift_x[k] = 0.0f;
      plan->shift_y[k] = 0.0f;
   }
   plan->aligned         = 0;
   plan->fallback_reason = reason;
   plan->mean_sad        = 0.0f;
   plan->align_error     = -1.0f;
   plan->trusted_blocks  = 0;
}

STBMEDEF stbme_plan *stbme_align(const unsigned char * const *frames, int n,
                                 int w, int h, int comp, const stbme_config *cfg_in)
{
   stbme_config cfg;
   stbme_plan *plan;
   float *lum;
   int range, bs, nb_x, nb_y, nblocks, b, k;
   float *blk_dx, *blk_dy, *blk_sad;
   char *blk_trust;

   if (!stbme_validate(n, w, h, comp) || !frames)
      return NULL;
   for (k = 0; k < n; ++k)
      if (!frames[k]) return NULL;

   if (cfg_in) cfg = *cfg_in;
   else stbme_config_default(&cfg);
   range = (int) STBME_CLAMP(cfg.search_range, 1, STBME_MAX_SHIFT);
   bs    = (int) STBME_CLAMP(cfg.block_size, 4, 64);
   if (bs > STBME_MIN(w, h)) bs = STBME_MAX(4, STBME_MIN(w, h));
   cfg.search_range = range;
   cfg.block_size   = bs;
   if (!(cfg.weight_gamma > 0.0f))  cfg.weight_gamma = 1.0f;
   cfg.tone_strength  = STBME_CLAMP(cfg.tone_strength, 0.0f, 1.0f);
   if (!(cfg.fallback_sad > 0.0f))  cfg.fallback_sad = 0.10f;
   cfg.trust_fraction = STBME_CLAMP(cfg.trust_fraction, 0.0f, 1.0f);

   plan = (stbme_plan *) calloc(1, sizeof(stbme_plan));
   if (!plan) return NULL;
   plan->w = w; plan->h = h; plan->comp = comp; plan->n = n;
   plan->cfg = cfg;

   nb_x = w / bs;
   nb_y = h / bs;
   nblocks = nb_x * nb_y;
   plan->total_blocks = nblocks;

   // Single frame: nothing to align, report success with zero shifts.
   if (n == 1) {
      plan->aligned = 1;
      plan->fallback_reason = STBME_FB_NONE;
      plan->align_error = 0.0f;
      return plan;
   }

   // Image too small for even one block: explicit fallback.
   if (nblocks == 0) {
      stbme_apply_fallback(plan, STBME_FB_UNTRUSTED);
      return plan;
   }

   lum = stbme_make_luma(frames, n, w, h, comp);
   blk_dx = (float *) malloc((size_t) nblocks * sizeof(float));
   blk_dy = (float *) malloc((size_t) nblocks * sizeof(float));
   blk_sad = (float *) malloc((size_t) nblocks * sizeof(float));
   blk_trust = (char *) malloc((size_t) nblocks);
   if (!lum || !blk_dx || !blk_dy || !blk_sad || !blk_trust) {
      free(lum); free(blk_dx); free(blk_dy); free(blk_sad); free(blk_trust);
      stbme_apply_fallback(plan, STBME_FB_BAD_INPUT);
      return plan;
   }

   // Each non-reference frame is aligned independently: per-frame block
   // votes, then a robust median gives that frame's single global shift.
   // Any frame with too few trusted votes (or shifts pinned to the search
   // edge) invalidates the whole alignment and triggers an explicit
   // zero-shift fallback for every frame.
   plan->aligned = 1;
   plan->fallback_reason = STBME_FB_NONE;
   {
      float err_sum = 0, sad_sum = 0;
      int trust_total = 0;
      int any_bad = 0;

      for (k = 1; k < n && !any_bad; ++k) {
         const float *ref = lum;
         const float *frm = lum + (size_t) k * w * h;
         int ntr = 0;
         float gx, gy;

         for (b = 0; b < nblocks; ++b) {
            int bx = (b % nb_x) * bs;
            int by = (b / nb_x) * bs;
            int dx, dy, best_dx = 0, best_dy = 0;
            float best = 1.0f;
            float sx_m, sx_0, sx_p, sy_m, sy_p;

            for (dy = -range; dy <= range; ++dy) {
               for (dx = -range; dx <= range; ++dx) {
                  float s = stbme_block_sad(ref, frm, w, h, bs, bx, by, dx, dy);
                  if (s < best) { best = s; best_dx = dx; best_dy = dy; }
               }
            }

            // Sub-pixel: independent 1-D parabolic refinement around the min.
            sx_0 = stbme_block_sad(ref, frm, w, h, bs, bx, by, best_dx, best_dy);
            sx_m = (best_dx > -range)
                 ? stbme_block_sad(ref, frm, w, h, bs, bx, by, best_dx - 1, best_dy) : sx_0;
            sx_p = (best_dx < range)
                 ? stbme_block_sad(ref, frm, w, h, bs, bx, by, best_dx + 1, best_dy) : sx_0;
            sy_m = (best_dy > -range)
                 ? stbme_block_sad(ref, frm, w, h, bs, bx, by, best_dx, best_dy - 1) : sx_0;
            sy_p = (best_dy < range)
                 ? stbme_block_sad(ref, frm, w, h, bs, bx, by, best_dx, best_dy + 1) : sx_0;

            blk_dx[b] = (float) best_dx + stbme_parabola(sx_m, sx_0, sx_p);
            blk_dy[b] = (float) best_dy + stbme_parabola(sy_m, sx_0, sy_p);
            blk_sad[b] = best;
            blk_trust[b] = (best < cfg.fallback_sad)
                        && (best_dx > -range && best_dx < range)
                        && (best_dy > -range && best_dy < range);
            if (blk_trust[b]) ++ntr;
         }

         if ((float) ntr < cfg.trust_fraction * (float) nblocks) {
            any_bad = 1;
            break;
         }

         {
            float *tx = (float *) malloc((size_t) ntr * sizeof(float));
            float *ty = (float *) malloc((size_t) ntr * sizeof(float));
            int j = 0;
            if (!tx || !ty) {
               free(tx); free(ty);
               free(lum); free(blk_dx); free(blk_dy); free(blk_sad); free(blk_trust);
               stbme_apply_fallback(plan, STBME_FB_BAD_INPUT);
               return plan;
            }
            for (b = 0; b < nblocks; ++b) {
               if (blk_trust[b]) {
                  tx[j] = blk_dx[b];
                  ty[j] = blk_dy[b];
                  sad_sum += blk_sad[b];
                  ++j;
               }
            }
            stbme_insertion_sort(tx, ntr);
            stbme_insertion_sort(ty, ntr);
            gx = (ntr & 1) ? tx[ntr / 2]
                           : 0.5f * (tx[ntr / 2 - 1] + tx[ntr / 2]);
            gy = (ntr & 1) ? ty[ntr / 2]
                           : 0.5f * (ty[ntr / 2 - 1] + ty[ntr / 2]);
            plan->shift_x[k] = gx;
            plan->shift_y[k] = gy;

            for (b = 0; b < nblocks; ++b) {
               if (blk_trust[b]) {
                  float ex = blk_dx[b] - gx, ey = blk_dy[b] - gy;
                  err_sum += (float) sqrt((double) ex * ex + (double) ey * ey);
               }
            }
            trust_total += ntr;
            free(tx);
            free(ty);
         }
      }

      if (any_bad) {
         stbme_apply_fallback(plan, STBME_FB_UNTRUSTED);
      } else {
         plan->trusted_blocks  = trust_total;
         plan->mean_sad        = sad_sum / (float) trust_total;
         plan->align_error     = err_sum / (float) trust_total;
      }
   }

   free(lum);
   free(blk_dx);
   free(blk_dy);
   free(blk_sad);
   free(blk_trust);
   return plan;
}

STBMEDEF void stbme_compute_weights(const stbme_plan *plan,
                                    const unsigned char * const *frames,
                                    float *weights)
{
   int w = plan->w, h = plan->h, comp = plan->comp, n = plan->n;
   int p, k;
   float sigma_q, sigma_g;
   float *lum, *warped;

   if (!plan || !frames || !weights) return;

   lum = stbme_make_luma(frames, n, w, h, comp);
   warped = (float *) malloc((size_t) n * w * h * sizeof(float));
   if (!lum || !warped) {
      free(lum);
      free(warped);
      return;
   }

   // Mid-tone well: gamma shrinks the Gaussian sigma, sharpening the curve.
   sigma_q = 127.5f / STBME_MAX(plan->cfg.weight_gamma, 1e-3f);
   sigma_g = 24.0f; // ghost rejection soft threshold (luma levels)

   for (k = 0; k < n; ++k) {
      const float *src = lum + (size_t) k * w * h;
      float *dst = warped + (size_t) k * w * h;
      float sx = plan->shift_x[k], sy = plan->shift_y[k];
      for (p = 0; p < h * w; ++p) {
         int x = p % w, y = p / w;
         dst[p] = stbme_samplef(src, w, h,
                                (float) x - sx, (float) y - sy);
      }
   }

   for (p = 0; p < w * h; ++p) {
      float ref_y = warped[p];
      float wsum = 0.0f;
      for (k = 0; k < n; ++k) {
         float yv = warped[(size_t) k * w * h + p];
         float dq = yv - 127.5f;
         float quality = (float) exp(-(double) (dq * dq) / (2.0 * sigma_q * sigma_q));
         float dg = yv - ref_y;
         float ghost = (k == 0)
                     ? 1.0f
                     : (float) exp(-(double) (dg * dg) / (2.0 * sigma_g * sigma_g));
         float wv = quality * ghost;
         weights[(size_t) k * w * h + p] = wv;
         wsum += wv;
      }
      if (wsum <= 1e-6f) {
         for (k = 0; k < n; ++k)
            weights[(size_t) k * w * h + p] = 1.0f / (float) n;
      } else {
         for (k = 0; k < n; ++k)
            weights[(size_t) k * w * h + p] /= wsum;
      }
   }

   free(lum);
   free(warped);
}

STBMEDEF void stbme_fuse(const stbme_plan *plan,
                         const unsigned char * const *frames,
                         const float *weights, unsigned char *out)
{
   int w = plan->w, h = plan->h, comp = plan->comp, n = plan->n;
   int p, c, k;
   float s = STBME_CLAMP(plan->cfg.tone_strength, 0.0f, 1.0f);

   if (!plan || !frames || !weights || !out) return;

   for (p = 0; p < w * h; ++p) {
      int x = p % w, y = p / w;
      for (c = 0; c < comp; ++c) {
         float acc = 0.0f, wsum = 0.0f, t, mapped;
         for (k = 0; k < n; ++k) {
            float wv = weights[(size_t) k * w * h + p];
            float sx = plan->shift_x[k], sy = plan->shift_y[k];
            unsigned char v = stbme_samplec(frames[k], comp, w, h,
                                            (float) x - sx,
                                            (float) y - sy, c);
            acc += wv * (float) v;
            wsum += wv;
         }
         if (wsum <= 1e-6f) wsum = 1.0f;
         t = acc / (wsum * 255.0f);
         if (t < 0) t = 0; else if (t > 1) t = 1;
         // Lerp between identity and smoothstep S-curve; endpoints fixed.
         mapped = (1.0f - s) * t + s * (3.0f * t * t - 2.0f * t * t * t);
         out[(size_t) p * comp + c] =
            (unsigned char) STBME_CLAMP(mapped * 255.0f + 0.5f, 0.0f, 255.0f);
      }
   }
}

STBMEDEF stbme_plan *stbme_blend(const unsigned char * const *frames, int n,
                                 int w, int h, int comp, const stbme_config *cfg,
                                 unsigned char *out)
{
   stbme_plan *plan;
   float *weights;

   if (!stbme_validate(n, w, h, comp) || !frames || !out)
      return NULL;
   plan = stbme_align(frames, n, w, h, comp, cfg);
   if (!plan) return NULL;

   weights = (float *) malloc((size_t) n * w * h * sizeof(float));
   if (!weights) {
      stbme_apply_fallback(plan, STBME_FB_BAD_INPUT);
      stbme_free_plan(plan);
      return NULL;
   }
   stbme_compute_weights(plan, frames, weights);
   stbme_fuse(plan, frames, weights, out);
   free(weights);
   return plan;
}

STBMEDEF void stbme_free_plan(stbme_plan *plan)
{
   free(plan);
}

#endif // STB_MULTIEXPOSE_IMPLEMENTATION

/*
------------------------------------------------------------------------------
This software is available under 2 licenses -- choose whichever you prefer.
------------------------------------------------------------------------------
ALTERNATIVE A - MIT License
Copyright (c) 2026
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
------------------------------------------------------------------------------
ALTERNATIVE B - Public Domain (www.unlicense.org)
This is free and unencumbered software released into the public domain.
Anyone is free to copy, modify, publish, use, compile, sell, or distribute this
software, either in source code form or as a compiled binary, for any purpose,
commercial or non-commercial, and by any means.
In jurisdictions that recognize copyright laws, the author or authors of this
software dedicate any and all copyright interest in the software to the public
domain. We make this dedication for the benefit of the public at large and to
the detriment of our heirs and successors. We intend this dedication to be an
overt act of relinquishment in perpetuity of all present and future rights to
this software under copyright law.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------
*/

#endif // INCLUDE_STB_MULTIEXPOSE_H
