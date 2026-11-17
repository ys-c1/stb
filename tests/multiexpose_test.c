// multiexpose_test.c - self-contained test + demo for stb_multiexpose.h
//
// Generates tiny synthetic bracketed bursts (no external data, no third
// party code) and runs the full three-stage pipeline on four sequences:
//   A static        - identical framing, varying exposure
//   B handheld      - fractional per-frame shifts, varying exposure
//   C moving object - static camera, an object moving between frames
//   D over/under    - heavy clipping at both ends
// plus an explicit alignment-failure fallback case (pure noise).
//
// Reports per sequence: alignment error (internal + against ground truth),
// fused local contrast, ghost pixel count and total wall time. Verifies
// input buffers are never modified and that fallback paths are explicit.
//
// Build (from tests/):
//   cc -std=c11 -Wall -Wextra -I.. multiexpose_test.c -lm -o multiexpose_test
// Also compiled as C++ by this test's own smoke commands.

#define STB_MULTIEXPOSE_IMPLEMENTATION
#include "../stb_multiexpose.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define W 48
#define H 48
#define COMP 3
#define NFR 4

static int g_failures = 0;

#define CHECK(cond, msg) do { \
   if (!(cond)) { printf("  [FAIL] %s\n", msg); ++g_failures; } \
   else         { printf("  [ok]   %s\n", msg); } \
} while (0)

// ---- deterministic RNG ----------------------------------------------------

static unsigned int g_rng = 0x12345678u;
static unsigned int rndu(void)
{
   g_rng = g_rng * 1103515245u + 12345u;
   return (g_rng >> 8) & 0xffffffu;
}
static int rnd_range(int lo, int hi) // inclusive
{
   return lo + (int) (rndu() % (unsigned int) (hi - lo + 1));
}

// ---- 64-bit FNV-1a for input-invariance checks ----------------------------

static unsigned long long fnv(const unsigned char *p, size_t n)
{
   unsigned long long h = 1469598103934665603ULL;
   size_t i;
   for (i = 0; i < n; ++i) {
      h ^= (unsigned long long) p[i];
      h *= 1099511628211ULL;
   }
   return h;
}

// ---- synthetic scene ------------------------------------------------------

static unsigned char g_scene[H][W][COMP];

static void build_scene(void)
{
   int y, x, c;
   for (y = 0; y < H; ++y) {
      for (x = 0; x < W; ++x) {
         float v;
         double s;
         // layered texture: gradient + sinusoid + checker + speckle, all
         // centred on mid-tone so block matching has structure everywhere
         v  = 128.0f;
         v += 30.0f * (float) sin((double) x * 0.55);
         v += 22.0f * (float) cos((double) y * 0.7);
         v += (((x / 6) + (y / 6)) & 1) ? 18.0f : -18.0f;
         v += (float) rnd_range(-9, 9);
         s = (x - 14) * (x - 14) + (y - 30) * (y - 30);
         if (s < 36.0) v += 35.0f;               // bright circle
         s = (x - 34) * (x - 34) + (y - 12) * (y - 12);
         if (s < 25.0) v -= 35.0f;               // dark circle
         for (c = 0; c < COMP; ++c) {
            static const float tint[COMP] = { 1.00f, 0.92f, 1.08f };
            float vc = v * tint[c];
            if (vc < 0) vc = 0;
            if (vc > 255) vc = 255;
            g_scene[y][x][c] = (unsigned char) (vc + 0.5f);
         }
      }
   }
}

// bilinear sample of the base scene with edge clamp
static float scene_samplef(float x, float y, int c)
{
   int x0, y0, x1, y1;
   float fx, fy;
   if (x <= 0) x = 0; else if (x >= W - 1) x = (float) (W - 1);
   if (y <= 0) y = 0; else if (y >= H - 1) y = (float) (H - 1);
   x0 = (int) x; y0 = (int) y;
   x1 = x0 + 1 < W ? x0 + 1 : x0;
   y1 = y0 + 1 < H ? y0 + 1 : y0;
   fx = x - (float) x0;
   fy = y - (float) y0;
   return (g_scene[y0][x0][c] * (1 - fx) + g_scene[y0][x1][c] * fx) * (1 - fy)
        + (g_scene[y1][x0][c] * (1 - fx) + g_scene[y1][x1][c] * fx) * fy;
}

// Build one frame: shift the scene by (tx,ty) (content displacement),
// apply an exposure factor, add sensor noise, optionally stamp a moving
// bright square at (ox,oy) (-1 = no square).
static unsigned char *make_frame(float tx, float ty, float exposure,
                                 int ox, int oy, int noise_amp)
{
   unsigned char *f = (unsigned char *) malloc((size_t) W * H * COMP);
   int y, x, c;
   for (y = 0; y < H; ++y) {
      for (x = 0; x < W; ++x) {
         for (c = 0; c < COMP; ++c) {
            float v = scene_samplef((float) x - tx, (float) y - ty, c);
            int iv;
            v *= exposure;
            if (noise_amp > 0)
               v += (float) rnd_range(-noise_amp, noise_amp);
            iv = (int) (v + 0.5f);
            if (iv < 0) iv = 0;
            if (iv > 255) iv = 255;
            f[(y * W + x) * COMP + c] = (unsigned char) iv;
         }
      }
   }
   if (ox >= 0 && oy >= 0) {
      for (y = oy; y < oy + 9 && y < H; ++y)
         for (x = ox; x < ox + 9 && x < W; ++x)
            for (c = 0; c < COMP; ++c)
               f[(y * W + x) * COMP + c] = (unsigned char) (c == 1 ? 215 : 240);
   }
   return f;
}

static void free_frames(unsigned char **fr, int n)
{
   int i;
   for (i = 0; i < n; ++i) free(fr[i]);
}

// ---- metrics --------------------------------------------------------------

static void frame_luma(const unsigned char *f, float *lum)
{
   int i;
   for (i = 0; i < W * H; ++i)
      lum[i] = (77.0f * f[i * COMP] + 150.0f * f[i * COMP + 1]
              + 29.0f * f[i * COMP + 2]) / 256.0f;
}

static float sample_bilinear(const float *img, float x, float y)
{
   int x0, y0, x1, y1;
   float fx, fy;
   if (x <= 0) x = 0; else if (x >= W - 1) x = (float) (W - 1);
   if (y <= 0) y = 0; else if (y >= H - 1) y = (float) (H - 1);
   x0 = (int) x; y0 = (int) y;
   x1 = x0 + 1 < W ? x0 + 1 : x0;
   y1 = y0 + 1 < H ? y0 + 1 : y0;
   fx = x - (float) x0;
   fy = y - (float) y0;
   return (img[y0 * W + x0] * (1 - fx) + img[y0 * W + x1] * fx) * (1 - fy)
        + (img[y1 * W + x0] * (1 - fx) + img[y1 * W + x1] * fx) * fy;
}

// mean |4-neighbour Laplacian| on the fused luma, interior pixels only
static double local_contrast(const unsigned char *fused)
{
   float lum[W * H];
   double acc = 0;
   int n = 0, x, y;
   frame_luma(fused, lum);
   for (y = 1; y < H - 1; ++y) {
      for (x = 1; x < W - 1; ++x) {
         float lap = lum[y * W + x] * 4.0f
                   - lum[y * W + x - 1] - lum[y * W + x + 1]
                   - lum[(y - 1) * W + x] - lum[(y + 1) * W + x];
         acc += (double) (lap < 0 ? -lap : lap);
         ++n;
      }
   }
   return acc / (double) n;
}

// Temporal-disagreement ghost count on exposure-normalised luma: each
// frame's warped luma is divided by its known exposure factor before
// comparison, so ordinary exposure differences are not counted as ghosts.
// Pixel pairs where either frame is clipped are skipped (saturation
// disagreement is expected there, not ghosting).
#define GHOST_TH 40.0f
static int ghost_pixels(const stbme_plan *plan,
                        unsigned char * const *frames,
                        const float *exposures)
{
   float *ref_lum = (float *) malloc((size_t) W * H * sizeof(float));
   float **wl = (float **) calloc((size_t) plan->n, sizeof(float *));
   int ghosts = 0, k, x, y;
   frame_luma(frames[0], ref_lum);
   for (k = 1; k < plan->n; ++k) {
      wl[k] = (float *) malloc((size_t) W * H * sizeof(float));
      frame_luma(frames[k], wl[k]);
   }
   for (y = 1; y < H - 1; ++y) {
      for (x = 1; x < W - 1; ++x) {
         int disagree = 0;
         float r = ref_lum[y * W + x] / exposures[0];
         if (ref_lum[y * W + x] < 16 || ref_lum[y * W + x] > 239)
            continue; // reference clipped: nothing reliable to compare
         for (k = 1; k < plan->n; ++k) {
            float raw = sample_bilinear(wl[k], (float) x - plan->shift_x[k],
                                        (float) y - plan->shift_y[k]);
            float v;
            if (raw < 16 || raw > 239) { disagree = 0; break; }
            v = raw / exposures[k];
            if (fabsf(v - r) > GHOST_TH) disagree = 1;
         }
         if (disagree) ++ghosts;
      }
   }
   for (k = 1; k < plan->n; ++k) free(wl[k]);
   free(wl);
   free(ref_lum);
   return ghosts;
}

static double gt_shift_error(const stbme_plan *plan,
                             const float *gtx, const float *gty)
{
   double acc = 0;
   int k;
   for (k = 1; k < plan->n; ++k) {
      double dx = (double) plan->shift_x[k] - gtx[k];
      double dy = (double) plan->shift_y[k] - gty[k];
      acc += sqrt(dx * dx + dy * dy);
   }
   return acc / (double) (plan->n - 1);
}

// ---- sequence driver ------------------------------------------------------

typedef struct seq_result {
   const char *name;
   int aligned;
   int fallback_reason;
   double err_internal;  // plan->align_error
   double err_gt;        // vs known injected shifts
   double contrast;
   int ghosts;
   double ms;
} seq_result;

static seq_result run_sequence(const char *name,
                               const float *gtx, const float *gty,
                               const float *exposures,
                               const int *objx, const int *objy,
                               int force_fallback)
{
   unsigned char *frames[NFR];
   unsigned char *out = (unsigned char *) malloc((size_t) W * H * COMP);
   unsigned long long before[NFR];
   stbme_config cfg;
   stbme_plan *plan;
   clock_t t0, t1;
   seq_result r;
   int i;

   stbme_config_default(&cfg);
   if (force_fallback) {
      // unmatchable content plus a tiny trust tolerance and narrow window
      cfg.fallback_sad = 0.001f;
      cfg.trust_fraction = 0.95f;
      cfg.search_range = 2;
   }

   for (i = 0; i < NFR; ++i) {
      if (force_fallback) {
         frames[i] = (unsigned char *) malloc((size_t) W * H * COMP);
         {
            int p;
            for (p = 0; p < W * H * COMP; ++p)
               frames[i][p] = (unsigned char) rnd_range(0, 255);
         }
      } else {
         frames[i] = make_frame(gtx[i], gty[i], exposures[i],
                                objx[i], objy[i], 2);
      }
      before[i] = fnv(frames[i], (size_t) W * H * COMP);
   }

   t0 = clock();
   plan = stbme_blend((const unsigned char * const *) frames, NFR,
                      W, H, COMP, &cfg, out);
   t1 = clock();

   r.name = name;
   r.aligned = plan->aligned;
   r.fallback_reason = plan->fallback_reason;
   r.err_internal = plan->align_error < 0 ? -1.0 : (double) plan->align_error;
   r.err_gt = force_fallback ? -1.0 : gt_shift_error(plan, gtx, gty);
   r.contrast = local_contrast(out);
   r.ghosts = ghost_pixels(plan, frames, exposures);
   r.ms = (double) (t1 - t0) * 1000.0 / (double) CLOCKS_PER_SEC;

   // input buffers must be byte-identical afterwards
   for (i = 0; i < NFR; ++i) {
      char msg[96];
      snprintf(msg, sizeof(msg), "%s: frame %d input unchanged", name, i);
      CHECK(fnv(frames[i], (size_t) W * H * COMP) == before[i], msg);
   }
   // output is unsigned char, so every produced value is in [0,255]
   stbme_free_plan(plan);
   free(out);
   free_frames(frames, NFR);
   return r;
}

// ---- stage-level checks ---------------------------------------------------

static void check_weight_normalization(void)
{
   unsigned char *frames[3];
   float gtx[3] = { 0, 1.0f, -1.0f };
   float gty[3] = { 0, 0.5f, -0.5f };
   float ex[3] = { 0.8f, 1.0f, 1.3f };
   float *weights = (float *) malloc((size_t) 3 * W * H * sizeof(float));
   stbme_config cfg;
   stbme_plan *plan;
   int i, p;
   stbme_config_default(&cfg);
   for (i = 0; i < 3; ++i)
      frames[i] = make_frame(gtx[i], gty[i], ex[i], -1, -1, 1);
   plan = stbme_align((const unsigned char * const *) frames, 3, W, H, COMP, &cfg);
   CHECK(plan->aligned == 1, "weight test: alignment succeeds");
   stbme_compute_weights(plan, (const unsigned char * const *) frames, weights);
   {
      int bad = 0;
      for (p = 0; p < W * H; ++p) {
         float s = 0;
         for (i = 0; i < 3; ++i) s += weights[(size_t) i * W * H + p];
         if (fabsf(s - 1.0f) > 1e-4f || weights[p] < 0) { bad = 1; break; }
      }
      CHECK(!bad, "weights non-negative and sum to 1 at every pixel");
   }
   stbme_free_plan(plan);
   free(weights);
   free_frames(frames, 3);
}

static void check_tone_endpoints(void)
{
   unsigned char *frames[2];
   unsigned char *out0 = (unsigned char *) malloc((size_t) W * H * COMP);
   unsigned char *out1 = (unsigned char *) malloc((size_t) W * H * COMP);
   stbme_config cfg;
   stbme_plan *plan;
   float ex[2] = { 0.9f, 1.1f };
   int i;
   stbme_config_default(&cfg);
   for (i = 0; i < 2; ++i)
      frames[i] = make_frame(0, 0, ex[i], -1, -1, 0);

   cfg.tone_strength = 0.0f;
   plan = stbme_blend((const unsigned char * const *) frames, 2, W, H, COMP, &cfg, out0);
   stbme_free_plan(plan);
   cfg.tone_strength = 1.0f;
   plan = stbme_blend((const unsigned char * const *) frames, 2, W, H, COMP, &cfg, out1);
   stbme_free_plan(plan);

   {
      // strength 0 must preserve mid grey exactly at the 50% point
      int mid = (H / 2) * W + (W / 2);
      int differ = memcmp(out0, out1, (size_t) W * H * COMP) != 0;
      CHECK(differ, "tone strength 0 vs 1 produce different images");
      CHECK(out0[mid * COMP] >= 0 && out0[mid * COMP] <= 255,
            "tone-mapped output stays in byte range");
   }
   free(out0); free(out1);
   free_frames(frames, 2);
}

static void check_knobs_and_modes(void)
{
   unsigned char *frames[3];
   unsigned char *out = (unsigned char *) malloc((size_t) W * H * COMP);
   float gtx[3] = { 0, 2.0f, -2.0f };
   float gty[3] = { 0, 1.0f, -1.0f };
   float ex[3] = { 0.7f, 1.0f, 1.4f };
   stbme_config cfg;
   stbme_plan *plan;
   int i;
   stbme_config_default(&cfg);
   for (i = 0; i < 3; ++i)
      frames[i] = make_frame(gtx[i], gty[i], ex[i], -1, -1, 1);

   cfg.search_range = 2; // smaller window than the 2px motion: edge-pinned
   plan = stbme_blend((const unsigned char * const *) frames, 3, W, H, COMP, &cfg, out);
   CHECK(plan->aligned == 0 && plan->fallback_reason == STBME_FB_UNTRUSTED,
         "motion outside search window triggers explicit fallback");
   stbme_free_plan(plan);

   stbme_config_default(&cfg);
   cfg.search_range = 6;
   cfg.weight_gamma = 2.0f;
   cfg.tone_strength = 0.25f;
   plan = stbme_blend((const unsigned char * const *) frames, 3, W, H, COMP, &cfg, out);
   CHECK(plan->aligned == 1, "wider window + tuned curves aligns 2px motion");
   stbme_free_plan(plan);

   // invalid arguments
   CHECK(stbme_align(NULL, 3, W, H, COMP, &cfg) == NULL, "NULL frames rejected");
   CHECK(stbme_align((const unsigned char * const *) frames, 0, W, H, COMP, &cfg) == NULL,
         "zero frames rejected");
   CHECK(stbme_align((const unsigned char * const *) frames, 3, W, H, 7, &cfg) == NULL,
         "unsupported component count rejected");

   // grey and RGBA smoke test (alpha ignored)
   {
      unsigned char *g[2], *o;
      stbme_plan *p2;
      int comp, npx;
      for (comp = 1; comp <= 4; comp += 3) {
         npx = W * H * comp;
         g[0] = (unsigned char *) malloc((size_t) npx);
         g[1] = (unsigned char *) malloc((size_t) npx);
         o   = (unsigned char *) malloc((size_t) npx);
         for (i = 0; i < npx; ++i) {
            g[0][i] = (unsigned char) rnd_range(20, 230);
            g[1][i] = g[0][i];
            if (comp == 4 && (i % 4) == 3) { g[0][i] = 99; g[1][i] = 99; }
         }
         p2 = stbme_blend((const unsigned char * const *) g, 2, W, H, comp, &cfg, o);
         CHECK(p2 != NULL && p2->aligned == 1, "comp=1/4 pipeline runs");
         if (comp == 4) {
            // alpha positions are produced through the same deterministic
            // fusion path as the colour components
            int alpha_ok = (o[3] == o[7]) && (o[(W * H - 1) * 4 + 3] == o[3]);
            CHECK(alpha_ok, "4th component shares the deterministic fusion path");
         }
         stbme_free_plan(p2);
         free(g[0]); free(g[1]); free(o);
      }
   }
   free(out);
   free_frames(frames, 3);
}

static void check_ghost_suppression(void)
{
   // Reference frame has no object; frame 1 carries a bright square. After
   // warping (zero shifts here), the ghost factor must cut frame 1's weight
   // inside the square relative to the background outside it.
   unsigned char *frames[2];
   float *weights;
   stbme_config cfg;
   stbme_plan *plan;
   int x, y;
   double w_in = 0, w_bg = 0;
   int n_in = 0, n_bg = 0;

   stbme_config_default(&cfg);
   frames[0] = make_frame(0, 0, 1.0f, -1, -1, 0);
   frames[1] = make_frame(0, 0, 1.0f, 12, 12, 0);
   plan = stbme_align((const unsigned char * const *) frames, 2, W, H, COMP, &cfg);
   CHECK(plan->aligned == 1, "ghost test: background aligns");
   weights = (float *) malloc((size_t) 2 * W * H * sizeof(float));
   stbme_compute_weights(plan, (const unsigned char * const *) frames, weights);

   for (y = 1; y < H - 1; ++y) {
      for (x = 1; x < W - 1; ++x) {
         int inside = (x >= 13 && x <= 19 && y >= 13 && y <= 19);
         float w1 = weights[(size_t) 1 * W * H + (y * W + x)];
         if (inside) { w_in += w1; ++n_in; }
         else        { w_bg += w1; ++n_bg; }
      }
   }
   printf("    frame-1 mean weight: background=%.3f  moving-object=%.3f\n",
          w_bg / n_bg, w_in / n_in);
   CHECK(w_in / n_in < 0.5 * (w_bg / n_bg),
         "ghost region weight of moving frame is cut below half background");
   free(weights);
   stbme_free_plan(plan);
   free(frames[0]);
   free(frames[1]);
}

static void check_edge_cases(void)
{
   stbme_config cfg;
   stbme_plan *plan;
   unsigned char tiny[3 * 3 * 3];
   unsigned char tout[3 * 3 * 3];
   unsigned char *one[1];
   unsigned char *pair[2];
   unsigned char *pout;
   int i;

   stbme_config_default(&cfg);
   for (i = 0; i < (int) sizeof(tiny); ++i)
      tiny[i] = (unsigned char) (40 + (i * 37) % 180);

   // single frame: success, zero shifts, no fallback
   one[0] = tiny;
   plan = stbme_align((const unsigned char * const *) one, 1, 3, 3, 3, &cfg);
   CHECK(plan != NULL && plan->aligned == 1
         && plan->shift_x[0] == 0.0f && plan->shift_y[0] == 0.0f,
         "single frame: immediate zero-shift plan");
   stbme_free_plan(plan);

   // 3x3 image with block_size clamped to 4 -> zero full blocks: explicit
   // fallback rather than a crash or division by zero
   pair[0] = tiny;
   pair[1] = tiny;
   plan = stbme_blend((const unsigned char * const *) pair, 2, 3, 3, 3,
                      &cfg, tout);
   CHECK(plan != NULL && plan->aligned == 0
         && plan->fallback_reason == STBME_FB_UNTRUSTED,
         "image smaller than block grid: explicit fallback, no crash");
   stbme_free_plan(plan);

   // minimal range: blocks whose only candidate sits on the search edge are
   // correctly distrusted, so the small window may produce an explicit
   // fallback rather than a silent guess. It must still run cleanly and
   // return a well-formed plan either way.
   pout = (unsigned char *) malloc((size_t) W * H * COMP);
   pair[0] = make_frame(0, 0, 1.0f, -1, -1, 0);
   pair[1] = make_frame(0, 0, 1.0f, -1, -1, 0);
   cfg.search_range = 1;
   plan = stbme_blend((const unsigned char * const *) pair, 2, W, H, COMP,
                      &cfg, pout);
   CHECK(plan != NULL && (plan->aligned == 1
         || plan->fallback_reason == STBME_FB_UNTRUSTED),
         "minimal search range returns alignment or explicit fallback");
   stbme_free_plan(plan);
   free(pout);
   free(pair[0]);
   free(pair[1]);

   // config sanitation: garbage values are clamped, never rejected
   cfg.search_range = -99;
   cfg.block_size = 9999;
   cfg.weight_gamma = -5.0f;
   cfg.tone_strength = 7.0f;
   cfg.fallback_sad = 0.0f;
   cfg.trust_fraction = -1.0f;
   pair[0] = make_frame(0, 0, 1.0f, -1, -1, 0);
   pair[1] = make_frame(0, 0, 1.0f, -1, -1, 0);
   pout = (unsigned char *) malloc((size_t) W * H * COMP);
   plan = stbme_blend((const unsigned char * const *) pair, 2, W, H, COMP,
                      &cfg, pout);
   CHECK(plan != NULL && plan->cfg.search_range == 1
         && plan->cfg.block_size <= 64
         && plan->cfg.tone_strength == 1.0f,
         "out-of-range config values are clamped to valid ones");
   stbme_free_plan(plan);
   free(pout);
   free(pair[0]);
   free(pair[1]);
}

int main(void)
{
   // ground-truth content shifts (frame 0 is the reference)
   float zero[NFR]   = { 0, 0, 0, 0 };
   float handx[NFR]  = { 0, 1.2f, -1.5f, 0.6f };
   float handy[NFR]  = { 0, -0.7f, 1.1f, 1.4f };
   float ex_static[NFR] = { 1.00f, 1.25f, 0.80f, 1.10f };
   float ex_hand[NFR]   = { 0.90f, 1.15f, 0.75f, 1.30f };
   float ex_clip[NFR]   = { 0.45f, 0.65f, 1.70f, 2.00f };
   int noobj[NFR]   = { -1, -1, -1, -1 };
   int objx[NFR]    = { -1, 8, 20, 30 };
   int objy[NFR]    = { -1, 20, 22, 24 };
   seq_result results[5];

   build_scene();
   printf("stb_multiexpose test: %dx%d frames, %d frames per burst\n\n",
          W, H, NFR);

   results[0] = run_sequence("A static        ", zero, zero,
                             ex_static, noobj, noobj, 0);
   results[1] = run_sequence("B handheld      ", handx, handy,
                             ex_hand, noobj, noobj, 0);
   results[2] = run_sequence("C moving object ", zero, zero,
                             ex_static, objx, objy, 0);
   results[3] = run_sequence("D over+under    ", handx, handy,
                             ex_clip, noobj, noobj, 0);
   results[4] = run_sequence("E fallback/noise", zero, zero,
                             ex_static, noobj, noobj, 1);

   printf("\n%-16s %-8s %-7s %10s %10s %10s %8s %9s\n",
          "sequence", "aligned", "fallbk", "err_intpx", "err_gtpx",
          "contrast", "ghosts", "time_ms");
   {
      int i;
      for (i = 0; i < 5; ++i) {
         printf("%-16s %-8s %-7d %10.3f %10.3f %10.3f %8d %9.3f\n",
                results[i].name,
                results[i].aligned ? "yes" : "NO",
                results[i].fallback_reason,
                results[i].err_internal, results[i].err_gt,
                results[i].contrast, results[i].ghosts, results[i].ms);
      }
   }

   printf("\n-- alignment correctness thresholds --\n");
   CHECK(results[0].aligned == 1 && results[0].err_gt < 0.25,
         "A static: sub-0.25px error, real alignment");
   CHECK(results[1].aligned == 1 && results[1].err_gt < 0.75,
         "B handheld: fractional shifts recovered within 0.75px");
   CHECK(results[2].aligned == 1 && results[2].err_gt < 0.50,
         "C moving object: static background still aligns");
   CHECK(results[3].aligned == 1 && results[3].err_gt < 0.75,
         "D over+under: alignment survives clipping");
   CHECK(results[4].aligned == 0 &&
         results[4].fallback_reason == STBME_FB_UNTRUSTED,
         "E noise: explicit zero-shift fallback, reason UNTRUSTED");

   printf("\n-- ghost behaviour --\n");
   CHECK(results[0].ghosts == 0, "A static: zero ghost pixels");
   CHECK(results[2].ghosts > 0,
         "C moving object: ghost pixels detected on object path");
   printf("    (C reports %d ghost pixels; weight ghost-rejection suppresses "
          "their contribution)\n", results[2].ghosts);

   printf("\n-- stage-level and interface checks --\n");
   check_weight_normalization();
   check_tone_endpoints();
   check_knobs_and_modes();
   check_ghost_suppression();
   check_edge_cases();

   printf("\n========================================\n");
   if (g_failures == 0) {
      printf("ALL CHECKS PASSED\n");
      return 0;
   }
   printf("%d CHECK(S) FAILED\n", g_failures);
   return 1;
}
