/* stb_bilateral - v0.1 - public domain - http://nothings.org/stb
   Bilateral filtering for 8-bit grayscale images:
     - exact (brute-force) bilateral filter
     - real-time approximation via a 3D bilateral grid
       (Chen, Paris & Durand, "Real-time edge-aware image processing with
        the bilateral grid", SIGGRAPH 2007)

   no warranty implied; use at your own risk.

LICENSE
  See end of file for license information.

DOCUMENTATION
  This header has no external dependencies beyond the C standard library.

  Before #including,

      #define STB_BILATERAL_IMPLEMENTATION

  in the file that you want to have the implementation.

  Define STB_BILATERAL_STATIC before including to make all functions static.
  STBBG_MALLOC / STBBG_FREE can be overridden like the other stb libraries.

  Basic usage:

      unsigned char *out = malloc(w*h);
      stbbg_bilateral_exact_gray8(w, h, img, out, sigma_spatial, sigma_range);
      stbbg_bilateral_grid_gray8(w, h, img, out, &params, NULL);

  Both functions work on 8-bit grayscale pixels (intensity 0..255) and use
  clamp-to-edge boundary handling. sigma_range is measured on the same
  0..255 intensity scale.

  Bilateral grid in three steps (see stbbg_bilateral_grid_gray8):
    1. splat   - every pixel is trilinearly accumulated into a 3D grid
                 whose axes are (x, y, intensity); two volumes are stored,
                 a weighted value sum and a weight sum.
    2. blur    - a separable Gaussian blur runs along each of the three
                 grid axes, on both volumes.
    3. slice   - each pixel trilinearly interpolates the two volumes at its
                 own (x, y, intensity) position and divides value by weight.

  When grid_w or grid_h is 1 the spatial axis collapses. With both
  grid_w == 1 and grid_h == 1 the 3-D grid has no spatial resolution, so
  the function takes a dedicated, well-defined path: a pure spatial
  (range-independent) Gaussian blur of the image. No division can hit a
  zero denominator on that path or anywhere else - every grid spacing is
  a guarded division, and slice normalization falls back to the input
  value if a cell ever received zero weight.
*/

#ifndef INCLUDE_STB_BILATERAL_H
#define INCLUDE_STB_BILATERAL_H

#ifdef __cplusplus
extern "C" {
#endif

#ifndef STBBG_DEF
#ifdef STB_BILATERAL_STATIC
#define STBBG_DEF static
#else
#define STBBG_DEF extern
#endif
#endif

#ifndef STBBG_MALLOC
#include <stdlib.h>
#define STBBG_MALLOC(sz) malloc(sz)
#define STBBG_FREE(p)     free(p)
#endif

/* Exact bilateral filter (reference implementation).
 *
 * For every pixel, every neighbor q inside a window of radius
 * ceil(3*sigma_spatial) contributes
 *   w = exp(-|p-q|^2 / (2*sigma_spatial^2))
 *       * exp(-(I_p-I_q)^2 / (2*sigma_range^2))
 * and the output is the weighted average. Both Gaussians are evaluated
 * directly in double precision so the reference is accurate rather than
 * table-quantized (this is the baseline every PSNR is measured against).
 *
 * Returns 0 on success, -1 on bad arguments.
 */
STBBG_DEF int stbbg_bilateral_exact_gray8(int w, int h,
                                          const unsigned char *input,
                                          unsigned char *output,
                                          float sigma_spatial,
                                          float sigma_range);

/* Parameters for the bilateral-grid approximation.
 *
 *   grid_w, grid_h : number of spatial grid cells along x/y. Must be >= 1.
 *                    Values > image dimension are clamped implicitly.
 *   luma_levels    : number of intensity grid cells (z axis), 1..256.
 *   sigma_spatial  : spatial sigma in *image pixels*, > 0.
 *   sigma_range    : range sigma on the 0..255 intensity scale, > 0.
 *   blur_radius    : Gaussian blur radius in *grid cells*, applied along
 *                    every grid axis. 0 means "choose automatically": each
 *                    axis gets radius = ceil(2.5 * its own grid-space
 *                    sigma), clamped to that axis's cell count minus one,
 *                    so a fine axis is never blurred with a coarse one's
 *                    radius.
 *
 * The grid-space sigmas are derived automatically:
 *   grid sigma xy = sigma_spatial / ((image_size-1) / (grid_size-1))
 *   grid sigma z  = sigma_range  / (255 / (luma_levels - 1))
 * A single-cell axis gets sigma 0 and a unit (no-op) kernel. The special
 * case grid_w == grid_h == 1 bypasses the grid entirely (see header docs).
 */
typedef struct stbbg_grid_params
{
   int   grid_w;
   int   grid_h;
   int   luma_levels;
   float sigma_spatial;
   float sigma_range;
   int   blur_radius;
} stbbg_grid_params;

/* Bilateral-grid bilateral filter.
 *
 * Returns 0 on success, -1 on bad arguments or allocation failure.
 * On success *out_bytes (when non-NULL) receives the number of bytes of
 * scratch memory allocated internally (grid volumes + blur buffers).
 */
STBBG_DEF int stbbg_bilateral_grid_gray8(int w, int h,
                                         const unsigned char *input,
                                         unsigned char *output,
                                         const stbbg_grid_params *params,
                                         long long *out_bytes);

#ifdef __cplusplus
}
#endif

#endif /* INCLUDE_STB_BILATERAL_H */

#ifdef STB_BILATERAL_IMPLEMENTATION

#include <math.h>
#include <string.h>

#ifndef STBBG_ASSERT
#include <assert.h>
#define STBBG_ASSERT(x) assert(x)
#endif

/* ------------------------------------------------------------------ */
/* small shared helpers                                                */
/* ------------------------------------------------------------------ */

static int stbbg_clampi(int x, int lo, int hi)
{
   if (x < lo) return lo;
   if (x > hi) return hi;
   return x;
}

/* ------------------------------------------------------------------ */
/* exact bilateral filter                                              */
/* ------------------------------------------------------------------ */

/* The reference filter evaluates the two Gaussian weights directly with
 * double-precision exp() rather than a lookup table: a table spanning the
 * full intensity range is too coarse near the origin (where range weights
 * are ~1 and matter most), and accuracy of the reference is what every
 * downstream PSNR number is measured against. */
STBBG_DEF int stbbg_bilateral_exact_gray8(int w, int h,
                                          const unsigned char *input,
                                          unsigned char *output,
                                          float sigma_spatial,
                                          float sigma_range)
{
   int radius, i, j, dy, dx;
   double inv_ss2, inv_sr2;

   if (w <= 0 || h <= 0 || !input || !output) return -1;
   if (!(sigma_spatial > 0.0f) || !(sigma_range > 0.0f)) return -1;

   radius = (int) ceil(3.0 * sigma_spatial);
   inv_ss2 = 1.0 / ((double) sigma_spatial * sigma_spatial);
   inv_sr2 = 1.0 / ((double) sigma_range * sigma_range);

   for (j = 0; j < h; j++) {
      for (i = 0; i < w; i++) {
         int center = input[j * w + i];
         double acc = 0.0, wsum = 0.0;
         /* clamp-to-edge: neighbor *access* is clamped, but the spatial
          * Gaussian weight always uses the unclamped offset, so border
          * pixels do not get an artificially inflated weight. */
         for (dy = -radius; dy <= radius; dy++) {
            int yy = j + dy;
            if (yy < 0) yy = 0;
            else if (yy > h - 1) yy = h - 1;
            for (dx = -radius; dx <= radius; dx++) {
               int xx = i + dx;
               int d2 = dx * dx + dy * dy;
               int q, id;
               double ws, wr, wt;
               if (xx < 0) xx = 0;
               else if (xx > w - 1) xx = w - 1;
               q = input[yy * w + xx];
               id = q - center;
               if (id < 0) id = -id;
               ws = exp(-0.5 * (double) d2 * inv_ss2);
               wr = exp(-0.5 * (double)(id * id) * inv_sr2);
               wt = ws * wr;
               acc += (double) q * wt;
               wsum += wt;
            }
         }
         if (wsum <= 0.0) {
            output[j * w + i] = (unsigned char) center;
         } else {
            int ri = (int) floor(acc / wsum + 0.5);
            output[j * w + i] = (unsigned char) stbbg_clampi(ri, 0, 255);
         }
      }
   }
   return 0;
}

/* ------------------------------------------------------------------ */
/* bilateral grid                                                      */
/* ------------------------------------------------------------------ */

typedef struct
{
   int     gx, gy, gz;   /* cell counts per axis, all >= 1 */
   size_t  n;            /* gx * gy * gz */
   float  *value;        /* weighted intensity sums  */
   float  *weight;       /* weight sums              */
} stbbg_grid;

static size_t stbbg_idx(const stbbg_grid *g, int x, int y, int z)
{
   return (size_t)(z * g->gy + y) * (size_t) g->gx + (size_t) x;
}

/* Image-domain separable spatial Gaussian blur (clamp-to-edge), on the
 * 0..255 intensity scale. Used for the degenerate grid whose spatial axes
 * collapse to a single cell: a 1-node grid cannot represent a per-pixel
 * spatial kernel, so the well-defined result of a "pure spatial blur" is
 * computed directly. range_sigma is ignored on this path by definition. */
static int stbbg_spatial_gaussian(int w, int h, const unsigned char *input,
                                  unsigned char *output, float sigma,
                                  long long *out_bytes)
{
   int radius = (int) ceil(3.0 * sigma);
   int klen = 2 * radius + 1;
   float *kern, *buf;
   long long bytes = 0;
   int i, j, k;

   if (radius <= 0) { memcpy(output, input, (size_t) w * (size_t) h); return 0; }
   kern = (float *) STBBG_MALLOC(sizeof(float) * (size_t) klen);
   buf  = (float *) STBBG_MALLOC(sizeof(float) * (size_t) w * (size_t) h);
   if (!kern || !buf) { STBBG_FREE(kern); STBBG_FREE(buf); return -1; }
   bytes = (long long) sizeof(float) * (klen + w * h);

   {
      double s = 0.0;
      for (k = -radius; k <= radius; k++) {
         kern[k + radius] = (float) exp(-0.5 * (double) k * k /
                                        (double)(sigma * sigma));
         s += kern[k + radius];
      }
      for (k = 0; k < klen; k++) kern[k] /= (float) s;
   }
   /* horizontal pass input -> buf (weight uses unclamped offset) */
   for (j = 0; j < h; j++)
      for (i = 0; i < w; i++) {
         double acc = 0.0;
         for (k = -radius; k <= radius; k++) {
            int xx = i + k;
            if (xx < 0) xx = 0;
            else if (xx > w - 1) xx = w - 1;
            acc += (double) input[j * w + xx] * kern[k + radius];
         }
         buf[j * w + i] = (float) acc;
      }
   /* vertical pass buf -> output (weight uses unclamped offset) */
   for (j = 0; j < h; j++)
      for (i = 0; i < w; i++) {
         double acc = 0.0;
         for (k = -radius; k <= radius; k++) {
            int yy = j + k;
            if (yy < 0) yy = 0;
            else if (yy > h - 1) yy = h - 1;
            acc += (double) buf[yy * w + i] * kern[k + radius];
         }
         {
            int r = (int) floor(acc + 0.5);
            output[j * w + i] = (unsigned char) stbbg_clampi(r, 0, 255);
         }
      }
   if (out_bytes) *out_bytes = bytes;
   STBBG_FREE(buf);
   STBBG_FREE(kern);
   return 0;
}

/* separable Gaussian blur of one grid volume along axis `axis`:
 *   axis 0 = x, 1 = y, 2 = z. Clamp-to-edge boundaries.
 * Cells are visited in storage order by decomposing the linear index once;
 * neighbor offsets are pure stride additions (no per-neighbor multiply). */
static void stbbg_blur_axis(float *dst, const float *src, const stbbg_grid *g,
                            int axis, const float *kernel, int radius)
{
   int dim = axis == 0 ? g->gx : axis == 1 ? g->gy : g->gz;
   size_t stride = axis == 0 ? 1
                 : axis == 1 ? (size_t) g->gx
                 : (size_t) g->gx * (size_t) g->gy;
   size_t n = g->n;
   size_t i;

   if (dim <= 1 || radius <= 0) {
      if (dst != src) memcpy(dst, src, n * sizeof(float));
      return;
   }
   for (i = 0; i < n; i++) {
      int cx, cy, cz, c;
      size_t linestart;
      int lo, hi, k;
      double acc = 0.0;
      /* decompose linear index once */
      cx = (int)(i % (size_t) g->gx);
      cy = (int)((i / (size_t) g->gx) % (size_t) g->gy);
      cz = (int)(i / ((size_t) g->gx * (size_t) g->gy));
      c = axis == 0 ? cx : axis == 1 ? cy : cz;
      /* index of the first cell of this line (coordinate 0 on axis) */
      linestart = i - (size_t) c * stride;
      lo = c - radius;
      hi = c + radius;
      if (lo < 0) lo = 0;
      if (hi > dim - 1) hi = dim - 1;
      for (k = lo; k <= hi; k++)
         acc += (double) src[linestart + (size_t) k * stride]
                 * kernel[k - c + radius];
      dst[i] = (float) acc;
   }
}

STBBG_DEF int stbbg_bilateral_grid_gray8(int w, int h,
                                         const unsigned char *input,
                                         unsigned char *output,
                                         const stbbg_grid_params *params,
                                         long long *out_bytes)
{
   stbbg_grid grid;
   float *tmp = 0;
   float *kern_x = 0, *kern_y = 0, *kern_z = 0;
   int rad_x, rad_y, rad_z, k;
   float inv_dx, inv_dy, inv_dz;
   float sig_gx, sig_gy, sig_gz;
   long long bytes = 0;
   int i, j;

   if (out_bytes) *out_bytes = 0;
   if (w <= 0 || h <= 0 || !input || !output || !params) return -1;
   if (params->grid_w < 1 || params->grid_h < 1 ||
       params->luma_levels < 1 || params->luma_levels > 256)
      return -1;
   if (!(params->sigma_spatial > 0.0f) ||
       !(params->sigma_range > 0.0f))
      return -1;

   /* Degenerate configuration: both spatial axes are a single cell. The
    * 3-D grid has no spatial resolution, so the well-defined edge-aware
    * interpretation requested by the API contract is a pure spatial
    * (range-independent) Gaussian blur. This path also covers gz == 1 and
    * never performs any division that could hit a zero denominator. */
   if (params->grid_w == 1 && params->grid_h == 1) {
      return stbbg_spatial_gaussian(w, h, input, output,
                                    params->sigma_spatial, out_bytes);
   }

   grid.gx = params->grid_w;
   grid.gy = params->grid_h;
   grid.gz = params->luma_levels;
   /* size_t multiply guarded against overflow on absurd inputs */
   if (grid.gx > 0 && (size_t) grid.gy > (size_t) -1 / (size_t) grid.gx)
      return -1;
   grid.n = (size_t) grid.gx * (size_t) grid.gy;
   if (grid.gz > 0 && grid.n > (size_t) -1 / (size_t) grid.gz)
      return -1;
   grid.n *= (size_t) grid.gz;
   if (grid.n > (size_t) -1 / sizeof(float) / 2)
      return -1;

   grid.value = (float *) STBBG_MALLOC(grid.n * sizeof(float));
   grid.weight = (float *) STBBG_MALLOC(grid.n * sizeof(float));
   tmp = (float *) STBBG_MALLOC(grid.n * sizeof(float));
   if (!grid.value || !grid.weight || !tmp) goto fail;
   bytes += (long long)(grid.n * sizeof(float)) * 3;
   memset(grid.value, 0, grid.n * sizeof(float));
   memset(grid.weight, 0, grid.n * sizeof(float));

   /* ---- step 1: splat (trilinear, endpoint mapping) ----
    * Pixel p maps to grid node g = p * (G-1) / (N-1), so the four image
    * corners land exactly on grid corner nodes. This keeps clamp-to-edge
    * slicing unbiased at the image border. Every spacing is a guarded
    * division: a single-cell axis (G == 1, handled by the dedicated
    * spatial-Gaussian path above) would get spacing 0, so no denominator
    * can ever be zero. */
   inv_dx = grid.gx > 1 ? (float)(grid.gx - 1) / (float)(w - 1) : 0.0f;
   inv_dy = grid.gy > 1 ? (float)(grid.gy - 1) / (float)(h - 1) : 0.0f;
   inv_dz = grid.gz > 1 ? (float)(grid.gz - 1) / 255.0f          : 0.0f;

   for (j = 0; j < h; j++) {
      float gyf = (float) j * inv_dy;
      int y0 = (int) gyf;
      float ty = gyf - (float) y0;
      int y1 = y0 + 1 < grid.gy ? y0 + 1 : y0;
      for (i = 0; i < w; i++) {
         float gxf = (float) i * inv_dx;
         int x0 = (int) gxf;
         float tx = gxf - (float) x0;
         int x1 = x0 + 1 < grid.gx ? x0 + 1 : x0;
         int inten = input[j * w + i];
         float gzf = (float) inten * inv_dz;
         int z0 = (int) gzf;
         float tz = gzf - (float) z0;
         int z1 = z0 + 1 < grid.gz ? z0 + 1 : z0;
         float w000 = (1 - tx) * (1 - ty) * (1 - tz);
         float w100 =      tx  * (1 - ty) * (1 - tz);
         float w010 = (1 - tx) *      ty  * (1 - tz);
         float w110 =      tx  *      ty  * (1 - tz);
         float w001 = (1 - tx) * (1 - ty) *      tz;
         float w101 =      tx  * (1 - ty) *      tz;
         float w011 = (1 - tx) *      ty  *      tz;
         float w111 =      tx  *      ty  *      tz;
         float v = (float) inten;
         size_t c000 = stbbg_idx(&grid, x0, y0, z0);
         size_t c100 = stbbg_idx(&grid, x1, y0, z0);
         size_t c010 = stbbg_idx(&grid, x0, y1, z0);
         size_t c110 = stbbg_idx(&grid, x1, y1, z0);
         size_t c001 = stbbg_idx(&grid, x0, y0, z1);
         size_t c101 = stbbg_idx(&grid, x1, y0, z1);
         size_t c011 = stbbg_idx(&grid, x0, y1, z1);
         size_t c111 = stbbg_idx(&grid, x1, y1, z1);
         grid.value[c000] += v * w000; grid.weight[c000] += w000;
         grid.value[c100] += v * w100; grid.weight[c100] += w100;
         grid.value[c010] += v * w010; grid.weight[c010] += w010;
         grid.value[c110] += v * w110; grid.weight[c110] += w110;
         grid.value[c001] += v * w001; grid.weight[c001] += w001;
         grid.value[c101] += v * w101; grid.weight[c101] += w101;
         grid.value[c011] += v * w011; grid.weight[c011] += w011;
         grid.value[c111] += v * w111; grid.weight[c111] += w111;
      }
   }

   /* ---- step 2: separable Gaussian blur on both volumes ----
    * grid-space sigma = image sigma / pixels-per-cell, where under the
    * endpoint mapping a cell spans (N-1)/(G-1) image units. */
   {
      /* a single-cell spatial axis has no spacing; it gets a unit kernel */
      float ppc_x = grid.gx > 1 ? (float)(w - 1) / (float)(grid.gx - 1) : 1.0f;
      float ppc_y = grid.gy > 1 ? (float)(h - 1) / (float)(grid.gy - 1) : 1.0f;
      float ppc_z = grid.gz > 1 ? 255.0f / (float)(grid.gz - 1)          : 1.0f;
      sig_gx = grid.gx > 1 ? params->sigma_spatial / ppc_x : 0.0f;
      sig_gy = grid.gy > 1 ? params->sigma_spatial / ppc_y : 0.0f;
      sig_gz = grid.gz > 1 ? params->sigma_range    / ppc_z : 0.0f;
   }

   /* auto radius per axis: ceil(2.5 * grid-space sigma), each axis gets
    * its own kernel so a fine axis is never blurred with a coarse one's
    * radius. blur_radius > 0 forces one shared radius on every axis. */
   if (params->blur_radius > 0) {
      rad_x = rad_y = rad_z = params->blur_radius;
   } else {
      rad_x = (int) ceil(2.5 * sig_gx);
      rad_y = (int) ceil(2.5 * sig_gy);
      rad_z = (int) ceil(2.5 * sig_gz);
   }
   /* clamp each radius to what its axis can actually represent */
   rad_x = stbbg_clampi(rad_x, 0, grid.gx - 1);
   rad_y = stbbg_clampi(rad_y, 0, grid.gy - 1);
   rad_z = stbbg_clampi(rad_z, 0, grid.gz - 1);

   kern_x = (float *) STBBG_MALLOC(sizeof(float) * (size_t)(2 * rad_x + 1));
   kern_y = (float *) STBBG_MALLOC(sizeof(float) * (size_t)(2 * rad_y + 1));
   kern_z = (float *) STBBG_MALLOC(sizeof(float) * (size_t)(2 * rad_z + 1));
   if (!kern_x || !kern_y || !kern_z) goto fail;
   bytes += (long long) sizeof(float)
            * (2 * rad_x + 1 + 2 * rad_y + 1 + 2 * rad_z + 1);

   {
      /* build one normalized 1-D kernel per axis */
      struct { float *kern; int rad; float sig; } axes[3] = {
         { kern_x, rad_x, sig_gx },
         { kern_y, rad_y, sig_gy },
         { kern_z, rad_z, sig_gz },
      };
      int a;
      for (a = 0; a < 3; a++) {
         float *kern = axes[a].kern;
         int rad = axes[a].rad;
         float sig = axes[a].sig;
         double s = 0.0;
         if (rad <= 0) { kern[0] = 1.0f; continue; }
         for (k = -rad; k <= rad; k++) {
            float gv = sig > 0.0f
                       ? (float) exp(-0.5 * (double) k * k /
                                     (double)(sig * sig))
                       : (k == 0 ? 1.0f : 0.0f);
            kern[k + rad] = gv;
            s += gv;
         }
         for (k = 0; k < 2 * rad + 1; k++) kern[k] /= (float) s;
      }
   }

   /* blur each volume: x -> y -> z, ping-ponging through tmp */
   {
      float *vols[2] = { grid.value, grid.weight };
      int vi;
      for (vi = 0; vi < 2; vi++) {
         float *vol = vols[vi];
         stbbg_blur_axis(tmp, vol, &grid, 0, kern_x, rad_x);
         stbbg_blur_axis(vol, tmp, &grid, 1, kern_y, rad_y);
         stbbg_blur_axis(tmp, vol, &grid, 2, kern_z, rad_z);
         memcpy(vol, tmp, grid.n * sizeof(float));
      }
   }

   /* ---- step 3: slice (trilinear) and normalize ---- */
   for (j = 0; j < h; j++) {
      float gyf = (float) j * inv_dy;
      int y0 = (int) gyf;
      float ty = gyf - (float) y0;
      int y1 = y0 + 1 < grid.gy ? y0 + 1 : y0;
      for (i = 0; i < w; i++) {
         float gxf = (float) i * inv_dx;
         int x0 = (int) gxf;
         float tx = gxf - (float) x0;
         int x1 = x0 + 1 < grid.gx ? x0 + 1 : x0;
         int inten = input[j * w + i];
         float gzf = (float) inten * inv_dz;
         int z0 = (int) gzf;
         float tz = gzf - (float) z0;
         int z1 = z0 + 1 < grid.gz ? z0 + 1 : z0;
         float w000 = (1 - tx) * (1 - ty) * (1 - tz);
         float w100 =      tx  * (1 - ty) * (1 - tz);
         float w010 = (1 - tx) *      ty  * (1 - tz);
         float w110 =      tx  *      ty  * (1 - tz);
         float w001 = (1 - tx) * (1 - ty) *      tz;
         float w101 =      tx  * (1 - ty) *      tz;
         float w011 = (1 - tx) *      ty  *      tz;
         float w111 =      tx  *      ty  *      tz;
         float vsum =
            grid.value [stbbg_idx(&grid, x0, y0, z0)] * w000 +
            grid.value [stbbg_idx(&grid, x1, y0, z0)] * w100 +
            grid.value [stbbg_idx(&grid, x0, y1, z0)] * w010 +
            grid.value [stbbg_idx(&grid, x1, y1, z0)] * w110 +
            grid.value [stbbg_idx(&grid, x0, y0, z1)] * w001 +
            grid.value [stbbg_idx(&grid, x1, y0, z1)] * w101 +
            grid.value [stbbg_idx(&grid, x0, y1, z1)] * w011 +
            grid.value [stbbg_idx(&grid, x1, y1, z1)] * w111;
         float wsum =
            grid.weight[stbbg_idx(&grid, x0, y0, z0)] * w000 +
            grid.weight[stbbg_idx(&grid, x1, y0, z0)] * w100 +
            grid.weight[stbbg_idx(&grid, x0, y1, z0)] * w010 +
            grid.weight[stbbg_idx(&grid, x1, y1, z0)] * w110 +
            grid.weight[stbbg_idx(&grid, x0, y0, z1)] * w001 +
            grid.weight[stbbg_idx(&grid, x1, y0, z1)] * w101 +
            grid.weight[stbbg_idx(&grid, x0, y1, z1)] * w011 +
            grid.weight[stbbg_idx(&grid, x1, y1, z1)] * w111;
         int ri;
         float r;
         if (!(wsum > 0.0f)) {
            /* no pixel ever splatted here: fall back to the input value */
            output[j * w + i] = (unsigned char) inten;
            continue;
         }
         r = vsum / wsum;
         ri = (int) floor(r + 0.5f);
         output[j * w + i] = (unsigned char) stbbg_clampi(ri, 0, 255);
      }
   }

   if (out_bytes) *out_bytes = bytes;
   STBBG_FREE(kern_z);
   STBBG_FREE(kern_y);
   STBBG_FREE(kern_x);
   STBBG_FREE(tmp);
   STBBG_FREE(grid.weight);
   STBBG_FREE(grid.value);
   return 0;

fail:
   STBBG_FREE(kern_z);
   STBBG_FREE(kern_y);
   STBBG_FREE(kern_x);
   STBBG_FREE(tmp);
   STBBG_FREE(grid.weight);
   STBBG_FREE(grid.value);
   return -1;
}

#endif /* STB_BILATERAL_IMPLEMENTATION */

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
FITNESS FOR ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------
*/
