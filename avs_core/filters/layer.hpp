
// Basic utils in C++ language, to be included in both base and processor specific (e.g. avx2) source modules, where they can
// be optimized for the specific instruction set.
// yuv add, subtract, mul, lighten, darken
// Chroma placement helpers (calculate_effective_mask*, prepare_effective_mask_for_row*)
// are now in overlay/blend_common.h, pulled in via layer.h.

// ---------------------------------------------------------------------------
// Rowprep function selectors — defined by the including TU to inject SIMD variants.
// Default: C scalar functions from overlay/blend_common.h.
//
// LAYER_ROWPREP_FN      — spatial averaging (+ Overlay-style opacity baking when
//                          full_opacity=false; not used for Layer's level scaling).
//                          Used for the full_opacity=true path: returns spatial
//                          averages (or maskp directly for MASK444).
//
// LAYER_ROWPREP_LEVEL_FN — Layer-style baking: (avg * level + 1) >> bpp.
//                          Used for the full_opacity=false path.
//
// Including TU (e.g. layer_avx2.cpp) defines them before this #include:
//   #define LAYER_ROWPREP_FN       prepare_effective_mask_for_row_avx2
//   #define LAYER_ROWPREP_LEVEL_FN prepare_effective_mask_for_row_level_baked_avx2
// ---------------------------------------------------------------------------
#ifndef LAYER_ROWPREP_FN
#  define LAYER_ROWPREP_FN  prepare_effective_mask_for_row
#endif
#ifndef LAYER_ROWPREP_LEVEL_FN
#  define LAYER_ROWPREP_LEVEL_FN  prepare_effective_mask_for_row_level_baked
#endif

DISABLE_WARNING_PUSH
DISABLE_WARNING_UNREFERENCED_LOCAL_VARIABLE

// YUV(A) mul 8-16 bits
// when chroma is processed, one can use/not use source chroma,
// Only when use_alpha: maskMode defines mask generation for chroma planes
// When use_alpha == false maskMode ignored
//
// full_opacity == true : level >= (1<<bpp); rowprep returns spatial avg; alpha_mask = eff[x].
// full_opacity == false: rowprep bakes (avg*level+1)>>bpp; alpha_mask = eff[x] directly.
template<MaskMode maskMode, typename pixel_t, bool lessthan16bits, bool is_chroma, bool use_chroma, bool has_alpha, bool full_opacity>
static void layer_yuv_mul_c_inner(BYTE* dstp8, const BYTE* ovrp8, const BYTE* maskp8, int dst_pitch, int overlay_pitch, int mask_pitch, int width, int height, int level, int bits_per_pixel) {
  pixel_t* dstp = reinterpret_cast<pixel_t*>(dstp8);
  const pixel_t* ovrp = reinterpret_cast<const pixel_t*>(ovrp8);
  const pixel_t* maskp = reinterpret_cast<const pixel_t*>(maskp8);
  dst_pitch /= sizeof(pixel_t);
  overlay_pitch /= sizeof(pixel_t);
  mask_pitch /= sizeof(pixel_t);

  typedef typename std::conditional<lessthan16bits, int, int64_t>::type calc_t;

  if constexpr (sizeof(pixel_t) == 1)
    bits_per_pixel = 8; // make quasi constexpr
  else if constexpr (sizeof(pixel_t) == 2 && !lessthan16bits)
    bits_per_pixel = 16; // make quasi constexpr

  // Buffer: needed for subsampled spatial averaging, or MASK444+!full_opacity baking.
  std::vector<pixel_t> effective_mask_buffer;
  if constexpr (has_alpha && (maskMode != MASK444 || !full_opacity)) {
    effective_mask_buffer.resize(width);
  }

  for (int y = 0; y < height; ++y) {
    const pixel_t* effective_mask_ptr = nullptr;

    // Rowprep: full_opacity path returns spatial avg (or maskp for MASK444).
    //          !full_opacity path bakes (avg * level + 1) >> bpp via LAYER_ROWPREP_LEVEL_FN.
    if constexpr (has_alpha) {
      if constexpr (full_opacity)
        effective_mask_ptr = LAYER_ROWPREP_FN<maskMode, pixel_t, true>(maskp, mask_pitch, width, effective_mask_buffer);
      else
        effective_mask_ptr = LAYER_ROWPREP_LEVEL_FN<maskMode, pixel_t, false>(maskp, mask_pitch, width, effective_mask_buffer, level, bits_per_pixel);
    }

    // Main blending loop — alpha_mask is fully prepared by rowprep.
    for (int x = 0; x < width; ++x) {
      // has_alpha=true: rowprep baked level in (!full_opacity) or returned avg (full_opacity).
      // has_alpha=false: flat level, no mask.
      int alpha_mask = has_alpha ? (int)effective_mask_ptr[x] : level;

      // fixme: no rounding? (code from YUY2)
      // for mul: no.
      if constexpr (!is_chroma)
        dstp[x] = (pixel_t)(dstp[x] + ((((((calc_t)ovrp[x] * dstp[x]) >> bits_per_pixel) - dstp[x]) * alpha_mask) >> bits_per_pixel));
      else if constexpr (use_chroma) {
        // chroma mode + process chroma
        dstp[x] = (pixel_t)(dstp[x] + (((calc_t)(ovrp[x] - dstp[x]) * alpha_mask) >> bits_per_pixel));
        // U = U + ( ((Uovr - U)*level) >> 8 )
        // V = V + ( ((Vovr - V)*level) >> 8 )
      }
      else {
        // non-chroma mode + process chroma
        const int half = 1 << (bits_per_pixel - 1);
        dstp[x] = (pixel_t)(dstp[x] + (((calc_t)(half - dstp[x]) * (alpha_mask / 2)) >> bits_per_pixel));
        // U = U + ( ((128 - U)*(level/2)) >> 8 )
        // V = V + ( ((128 - V)*(level/2)) >> 8 )
      }
    }
    dstp += dst_pitch;
    ovrp += overlay_pitch;
    if constexpr (has_alpha) {
      if constexpr (maskMode == MASK420 || maskMode == MASK420_MPEG2 || maskMode == MASK420_TOPLEFT)
        maskp += mask_pitch * 2;
      else
        maskp += mask_pitch;
    }
  }
}

// Outer dispatcher: checks level >= (1<<bpp) at runtime to select full_opacity branch.
template<MaskMode maskMode, typename pixel_t, bool lessthan16bits, bool is_chroma, bool use_chroma, bool has_alpha>
static void layer_yuv_mul_c(BYTE* dstp8, const BYTE* ovrp8, const BYTE* maskp8, int dst_pitch, int overlay_pitch, int mask_pitch, int width, int height, int level, int bits_per_pixel) {
  if constexpr (!has_alpha) {
    // No mask — full_opacity choice is irrelevant; use true to skip dead buffer allocation.
    layer_yuv_mul_c_inner<maskMode, pixel_t, lessthan16bits, is_chroma, use_chroma, false, true>(
      dstp8, ovrp8, maskp8, dst_pitch, overlay_pitch, mask_pitch, width, height, level, bits_per_pixel);
  } else {
    if (level >= (1 << bits_per_pixel))
      layer_yuv_mul_c_inner<maskMode, pixel_t, lessthan16bits, is_chroma, use_chroma, true, true>(
        dstp8, ovrp8, maskp8, dst_pitch, overlay_pitch, mask_pitch, width, height, level, bits_per_pixel);
    else
      layer_yuv_mul_c_inner<maskMode, pixel_t, lessthan16bits, is_chroma, use_chroma, true, false>(
        dstp8, ovrp8, maskp8, dst_pitch, overlay_pitch, mask_pitch, width, height, level, bits_per_pixel);
  }
}

// YUV(A) mul 32 bits
template<MaskMode maskMode, bool is_chroma, bool use_chroma, bool has_alpha>
static void layer_yuv_mul_f_c(BYTE* dstp8, const BYTE* ovrp8, const BYTE* maskp8, int dst_pitch, int overlay_pitch, int mask_pitch, int width, int height, float opacity) {
  float* dstp = reinterpret_cast<float*>(dstp8);
  const float* ovrp = reinterpret_cast<const float*>(ovrp8);
  const float* maskp = reinterpret_cast<const float*>(maskp8);
  dst_pitch /= sizeof(float);
  overlay_pitch /= sizeof(float);
  mask_pitch /= sizeof(float);

  // precalculate mask buffer
  std::vector<float> effective_mask_buffer;
  if constexpr (has_alpha && maskMode != MASK444) {
    effective_mask_buffer.resize(width);
  }

  for (int y = 0; y < height; ++y) {
    const float* effective_mask_ptr = nullptr;

    // precalculate effective mask for this row
    if constexpr (has_alpha) {
      effective_mask_ptr = prepare_effective_mask_for_row_f<maskMode>(maskp, mask_pitch, width, effective_mask_buffer);
    }

    // Main blending loop - now simplified and vectorizable
    for (int x = 0; x < width; ++x) {
      float effective_mask = has_alpha ? effective_mask_ptr[x] : 0.0f;
      float alpha_mask = has_alpha ? effective_mask * opacity : opacity;

      if constexpr (!is_chroma)
        dstp[x] = dstp[x] + (ovrp[x] * dstp[x] - dstp[x]) * alpha_mask;
      else if constexpr (use_chroma) {
        // chroma mode + process chroma
        dstp[x] = dstp[x] + (ovrp[x] - dstp[x]) * alpha_mask;
        // U = U + ( ((Uovr - U)*level) >> 8 )
        // V = V + ( ((Vovr - V)*level) >> 8 )
      }
      else {
        // non-chroma mode + process chroma
        constexpr float half = 0.0f;
        dstp[x] = dstp[x] + (half - dstp[x]) * (alpha_mask * 0.5f);
        // U = U + ( ((128 - U)*(level/2)) >> 8 )
        // V = V + ( ((128 - V)*(level/2)) >> 8 )
      }
    }
    dstp += dst_pitch;
    ovrp += overlay_pitch;
    if constexpr (has_alpha) {
      if constexpr (maskMode == MASK420 || maskMode == MASK420_MPEG2 || maskMode == MASK420_TOPLEFT)
        maskp += mask_pitch * 2;
      else
        maskp += mask_pitch;
    }
  }
}

// YUV mul 8-16 bits
// when chroma is processed, one can use/not use source chroma
// Only when use_alpha: maskMode defines mask generation for chroma planes
// When use_alpha == false -> maskMode ignored


// Generated ASM analysis after doing chroma placement-dependent mask precalculation
// => vectorized blending now recognized!
//
// Refactoring separated mask calculation from blending, enabling compilers to auto-vectorize the main loop.
// Memory overhead: single row buffer (e.g., 1920 bytes for 1080p) fits comfortably in L1 cache.
// Estimated speedups were told by AI, but preparing the inputs for Layer takes time, i could not measure only the
// blending loop; but the speedup is significant (except 444 where no precalculation is needed)
//
// Mask calculation (e.g. MASK420_MPEG2 2x2 gather with sliding window):
//   - All compilers: Scalar with no or max. 2x unrolling (complex gather+dependency pattern blocks vectorization)
//   - Minimal overhead: ~1-2% of total runtime, one-time cost per row
//
// Main blending loop vectorization results:
//   MSVC 2022 (SSE4.1):  4-wide vectorization, ~80 instructions per 16 pixels
//     - Uses pmulld (SSE4.1) for 32-bit multiply, pmovzxbd for byte→dword extension
//     - Fallback path: 16→4→1 (main loop, then scalar cleanup)
//     - Estimated speedup: 3-4x vs scalar
//   
//   Intel C++ 2025 (SSE2):  16-wide vectorization, ~150 instructions per 16 pixels
//     - Workaround for missing pmulld: pmuludq + shuffle for odd/even dwords (complex!)
//     - Complex unpacking chain: 4× punpcklbw/punpckhbw + punpcklwd/punpckhwd for byte→dword
//     - Fallback path: 16→1 (main loop processes full 16, scalar cleanup for remainder)
//     - Estimated speedup: 6-8x vs scalar
//   
//   Intel C++ 2025 (AVX2):  16-wide vectorization (2×YMM), ~60 instructions per 16 pixels
//     - Native vpmulld on YMM, vpmovzxbd for clean extension, vpshufb+vpackusdw packing
//     - Fallback path: 16→4→1 (YMM main, XMM for 4-15 remaining, scalar for <4)
//     - Requires XMM6-XMM14 save/restore (ABI requirement), adds minimal function overhead
//     - Estimated speedup: 10-12x vs scalar
//   
//   Intel C++ 2025 (AVX-512):  16-wide vectorization (ZMM), ~40 instructions per 16 pixels
//     - Predicated execution via k-registers: vpcmpuq+kunpckbw for boundary checking
//     - Masked loads/stores (vmovdqu8 {k1}{z}) eliminate separate cleanup loops entirely!
//     - Native vpmovdb for efficient dword→byte packing, vpmovzxbd for extension
//     - Fallback path: 16→1 with masking (5+ remaining uses masked 16-wide, <5 uses scalar)
//     - No XMM register save overhead (ZMM registers are volatile), only vzeroupper at exit
//     - Estimated speedup: 12-15x vs scalar (requires Ice Lake+, Zen4+; may throttle on some CPUs)
//
// Sequential mask access pattern (vs. inline 2x2, 1x2, 2x3 gather) was critical for unlocking
// auto-vectorization. The 16-wide SIMD implementations process the same data in 1/10th the time
// despite the overhead of mask pre-calculation, proving the separation-of-concerns approach.
// 
// Instruction count <> performance; Intel SSE2 uses ~2x more instructions than 
// MSVC SSE4.1, but achieves better speedup due to:
//   1. Better instruction-level parallelism (ILP) - processes all 16 pixels in parallel
//   2. Lower loop overhead - single iteration vs 4 iterations for 16 pixels
//   3. Better memory bandwidth utilization - coalesced loads/stores
//   4. Aggressive register usage (all 16 XMM) reduces memory traffic
// The 16-wide approach's higher upfront cost is amortized by massive parallelism.

// Separated mask precalculation per row.
// full_opacity == true : level >= (1<<bpp); rowprep returns spatial avg; alpha_mask = eff[x].
// full_opacity == false: rowprep bakes (avg*level+1)>>bpp; alpha_mask = eff[x] directly.
// subtract=false only: overlay clip is pre-inverted by Layer::Create when op="Subtract".
template<MaskMode maskMode, typename pixel_t, bool lessthan16bits, bool is_chroma, bool use_chroma, bool has_alpha, bool full_opacity>
static void layer_yuv_add_c_inner(BYTE* dstp8, const BYTE* ovrp8, const BYTE* mask8, int dst_pitch, int overlay_pitch, int mask_pitch, int width, int height, int level, int bits_per_pixel) {
  pixel_t* dstp = reinterpret_cast<pixel_t*>(dstp8);
  const pixel_t* ovrp = reinterpret_cast<const pixel_t*>(ovrp8);
  const pixel_t* maskp = reinterpret_cast<const pixel_t*>(mask8);
  dst_pitch /= sizeof(pixel_t);
  overlay_pitch /= sizeof(pixel_t);
  mask_pitch /= sizeof(pixel_t);

  typedef typename std::conditional<lessthan16bits, int, int64_t>::type calc_t;

  if constexpr (sizeof(pixel_t) == 1)
    bits_per_pixel = 8; // make quasi constexpr
  else if constexpr (sizeof(pixel_t) == 2 && !lessthan16bits)
    bits_per_pixel = 16; // make quasi constexpr

  const int rounder = 1 << (bits_per_pixel - 1);

  // Buffer: needed for subsampled spatial averaging, or MASK444+!full_opacity baking.
  std::vector<pixel_t> effective_mask_buffer;
  if constexpr (has_alpha && (maskMode != MASK444 || !full_opacity)) {
    effective_mask_buffer.resize(width);
  }

  for (int y = 0; y < height; ++y) {
    const pixel_t* effective_mask_ptr = nullptr;

    // Rowprep: full_opacity path returns spatial avg (or maskp for MASK444).
    //          !full_opacity path bakes (avg * level + 1) >> bpp via LAYER_ROWPREP_LEVEL_FN.
    if constexpr (has_alpha) {
      if constexpr (full_opacity)
        effective_mask_ptr = LAYER_ROWPREP_FN<maskMode, pixel_t, true>(maskp, mask_pitch, width, effective_mask_buffer);
      else
        effective_mask_ptr = LAYER_ROWPREP_LEVEL_FN<maskMode, pixel_t, false>(maskp, mask_pitch, width, effective_mask_buffer, level, bits_per_pixel);
    }

    // Main blending loop — alpha_mask is fully prepared by rowprep.
    for (int x = 0; x < width; ++x) {
      // has_alpha=true: rowprep baked level in (!full_opacity) or returned avg (full_opacity).
      // has_alpha=false: flat level, no mask.
      int alpha_mask = has_alpha ? (int)effective_mask_ptr[x] : level;

      if constexpr (!is_chroma || use_chroma)
        dstp[x] = (pixel_t)(dstp[x] + (((calc_t)(ovrp[x] - dstp[x]) * alpha_mask + rounder) >> bits_per_pixel));
      else {
        const int half = 1 << (bits_per_pixel - 1);
        dstp[x] = (pixel_t)(dstp[x] + (((calc_t)(half - dstp[x]) * alpha_mask + rounder) >> bits_per_pixel));
      }
    }
    dstp += dst_pitch;
    ovrp += overlay_pitch;
    if constexpr (has_alpha) {
      if constexpr (maskMode == MASK420 || maskMode == MASK420_MPEG2 || maskMode == MASK420_TOPLEFT)
        maskp += mask_pitch * 2;
      else
        maskp += mask_pitch;
    }
  }
}

// Outer dispatcher: checks level >= (1<<bpp) at runtime to select full_opacity branch.
template<MaskMode maskMode, typename pixel_t, bool lessthan16bits, bool is_chroma, bool use_chroma, bool has_alpha>
static void layer_yuv_add_c(BYTE* dstp8, const BYTE* ovrp8, const BYTE* mask8, int dst_pitch, int overlay_pitch, int mask_pitch, int width, int height, int level, int bits_per_pixel) {
  if constexpr (!has_alpha) {
    // No mask — full_opacity choice is irrelevant; use true to skip dead buffer allocation.
    layer_yuv_add_c_inner<maskMode, pixel_t, lessthan16bits, is_chroma, use_chroma, false, true>(
      dstp8, ovrp8, mask8, dst_pitch, overlay_pitch, mask_pitch, width, height, level, bits_per_pixel);
  } else {
    if (level >= (1 << bits_per_pixel))
      layer_yuv_add_c_inner<maskMode, pixel_t, lessthan16bits, is_chroma, use_chroma, true, true>(
        dstp8, ovrp8, mask8, dst_pitch, overlay_pitch, mask_pitch, width, height, level, bits_per_pixel);
    else
      layer_yuv_add_c_inner<maskMode, pixel_t, lessthan16bits, is_chroma, use_chroma, true, false>(
        dstp8, ovrp8, mask8, dst_pitch, overlay_pitch, mask_pitch, width, height, level, bits_per_pixel);
  }
}

// YUV(A) add 32 bits — subtract is handled by pre-inverted overlay (Layer::Create).
template<MaskMode maskMode, bool is_chroma, bool use_chroma, bool has_alpha>
static void layer_yuv_add_f_c(BYTE* dstp8, const BYTE* ovrp8, const BYTE* maskp8, int dst_pitch, int overlay_pitch, int mask_pitch, int width, int height, float opacity) {
  float* dstp = reinterpret_cast<float*>(dstp8);
  const float* ovrp = reinterpret_cast<const float*>(ovrp8);
  const float* maskp = reinterpret_cast<const float*>(maskp8);
  dst_pitch /= sizeof(float);
  overlay_pitch /= sizeof(float);
  mask_pitch /= sizeof(float);

  // precalculate mask buffer
  std::vector<float> effective_mask_buffer;
  if constexpr (has_alpha && maskMode != MASK444) {
    effective_mask_buffer.resize(width);
  }

  for (int y = 0; y < height; ++y) {
    const float* effective_mask_ptr = nullptr;

    // precalculate effective mask for this row
    if constexpr (has_alpha) {
      if constexpr (maskMode == MASK444) {
        // Direct access to original mask - no pre-calculation needed
        effective_mask_ptr = maskp;
      }
      else {
        // Initialize sliding window state for MPEG2 modes
        float mask_right = 0.0f;
        if constexpr (maskMode == MASK420_MPEG2) {
          mask_right = maskp[0] + maskp[0 + mask_pitch];
        }
        else if constexpr (maskMode == MASK422_MPEG2) {
          mask_right = maskp[0];
        }

        // precalculate averaged mask values
        for (int x = 0; x < width; ++x) {
          effective_mask_buffer[x] = calculate_effective_mask_f<maskMode>(
            maskp, x, mask_pitch, mask_right
          );
        }
        effective_mask_ptr = effective_mask_buffer.data();
      }
    }

    // Main blending loop - now simplified and vectorizable
    for (int x = 0; x < width; ++x) {
      float effective_mask = has_alpha ? effective_mask_ptr[x] : 0.0f;
      float alpha_mask = has_alpha ? effective_mask * opacity : opacity;

      if constexpr (!is_chroma || use_chroma) {
        dstp[x] = dstp[x] + (ovrp[x] - dstp[x]) * alpha_mask;
      }
      else {
        constexpr float half = 0.0f;
        dstp[x] = dstp[x] + (half - dstp[x]) * alpha_mask;
      }
    }
    dstp += dst_pitch;
    ovrp += overlay_pitch;
    if constexpr (has_alpha) {
      if constexpr (maskMode == MASK420 || maskMode == MASK420_MPEG2 || maskMode == MASK420_TOPLEFT)
        maskp += mask_pitch * 2;
      else
        maskp += mask_pitch;
    }
  }
}

// Unlike RGBA version, YUVA does not update destination alpha
template<int mode, MaskMode maskMode, typename pixel_t, bool lessthan16bits, bool lumaonly, bool has_alpha>
static void layer_yuv_lighten_darken_c(
  BYTE* dstp8, BYTE* dstp8_u, BYTE* dstp8_v,/* BYTE* dstp8_a,*/
  const BYTE* ovrp8, const BYTE* ovrp8_u, const BYTE* ovrp8_v, const BYTE* maskp8,
  int dst_pitch, int dst_pitchUV,
  int overlay_pitch, int overlay_pitchUV,
  int mask_pitch,
  int width, int height, int level, int thresh,
  int bits_per_pixel) {

  pixel_t* dstp = reinterpret_cast<pixel_t*>(dstp8);
  pixel_t* dstp_u = reinterpret_cast<pixel_t*>(dstp8_u);
  pixel_t* dstp_v = reinterpret_cast<pixel_t*>(dstp8_v);
  // pixel_t* dstp_a = reinterpret_cast<pixel_t *>(dstp8_a); // not destination alpha update

  const pixel_t* ovrp = reinterpret_cast<const pixel_t*>(ovrp8);
  const pixel_t* ovrp_u = reinterpret_cast<const pixel_t*>(ovrp8_u);
  const pixel_t* ovrp_v = reinterpret_cast<const pixel_t*>(ovrp8_v);
  const pixel_t* maskp = reinterpret_cast<const pixel_t*>(maskp8);

  dst_pitch /= sizeof(pixel_t);
  dst_pitchUV /= sizeof(pixel_t);
  overlay_pitch /= sizeof(pixel_t);
  overlay_pitchUV /= sizeof(pixel_t);
  mask_pitch /= sizeof(pixel_t);

  const int cwidth = (maskMode == MASK444) ? width : (maskMode == MASK411) ? width >> 2 : width >> 1; // 444:/1  420,422:/2  411:/4
  const int cheight = (maskMode == MASK444 || maskMode == MASK422 || maskMode == MASK422_MPEG2 || maskMode == MASK422_TOPLEFT || maskMode == MASK411) ? height : height >> 1; // 444,422,411:/1  420:/2

  // In lighten/darken we need 3 buffers:
  std::vector<pixel_t> ovr_buffer;
  std::vector<pixel_t> src_buffer;
  std::vector<pixel_t> mask_buffer;

  // precalculate mask buffer et al.
  if constexpr (maskMode != MASK444) {
    ovr_buffer.resize(cwidth);
    src_buffer.resize(cwidth);
    if constexpr (has_alpha)
      mask_buffer.resize(cwidth);
  }

  using calc_t = typename std::conditional < lessthan16bits, int, int64_t>::type; // for non-overflowing 16 bit alpha_mul
  if constexpr (sizeof(pixel_t) == 1)
    bits_per_pixel = 8; // make quasi constexpr
  else if constexpr (sizeof(pixel_t) == 2 && !lessthan16bits)
    bits_per_pixel = 16; // make quasi constexpr

  const int rounder = 1 << (bits_per_pixel - 1);

  // for subsampled color spaces first do chroma, luma is only used for decision
  // second pass will do luma only
  for (int y = 0; y < cheight; ++y) {

    // Prepare all three pointers using the helper
    const pixel_t* ovr_ptr = prepare_effective_mask_for_row<maskMode, pixel_t>(ovrp, overlay_pitch, cwidth, ovr_buffer);
    const pixel_t* src_ptr = prepare_effective_mask_for_row<maskMode, pixel_t>(dstp, dst_pitch, cwidth, src_buffer);
    const pixel_t* effective_mask_ptr = nullptr;
    if constexpr (has_alpha) {
      effective_mask_ptr = prepare_effective_mask_for_row<maskMode, pixel_t>(maskp, mask_pitch, cwidth, mask_buffer);
    }

    for (int x = 0; x < cwidth; ++x) {
      int ovr = ovr_ptr[x];
      int src = src_ptr[x];
      int effective_mask = has_alpha ? effective_mask_ptr[x] : 0;

      const int alpha = has_alpha ? (int)(((calc_t)effective_mask * level + 1) >> bits_per_pixel) : level;

      int alpha_mask;
      if constexpr (mode == LIGHTEN)
        alpha_mask = ovr > (src + thresh) ? alpha : 0; // YUY2 was wrong: alpha_mask = (thresh + ovr) > src ? level : 0;
      else // DARKEN
        alpha_mask = ovr < (src - thresh) ? alpha : 0; // YUY2 was wrong: alpha_mask = (thresh + src) > ovr ? level : 0;

      if constexpr (!lumaonly)
      {
        // chroma u,v
        dstp_u[x] = dstp_u[x] + (int)(((calc_t)(ovrp_u[x] - dstp_u[x]) * alpha_mask + rounder) >> bits_per_pixel);
        dstp_v[x] = dstp_v[x] + (int)(((calc_t)(ovrp_v[x] - dstp_v[x]) * alpha_mask + rounder) >> bits_per_pixel);
      }

      // for 444: update here, width/height is the same as for chroma
      if constexpr (maskMode == MASK444)
        dstp[x] = dstp[x] + (int)(((calc_t)(ovrp[x] - dstp[x]) * alpha_mask + rounder) >> bits_per_pixel);
    }
    if constexpr (maskMode == MASK420 || maskMode == MASK420_MPEG2 || maskMode == MASK420_TOPLEFT) {
      dstp += dst_pitch * 2; // skip vertical subsampling
      ovrp += overlay_pitch * 2;
      if constexpr (has_alpha) {
        //dstp_a += dst_pitch * 2;
        maskp += mask_pitch * 2;
      }
    }
    else {
      dstp += dst_pitch;
      ovrp += overlay_pitch;
      if constexpr (has_alpha) {
        //dstp_a += dst_pitch;
        maskp += mask_pitch;
      }
    }

    if constexpr (!lumaonly) {
      dstp_u += dst_pitchUV;
      dstp_v += dst_pitchUV;
      ovrp_u += overlay_pitchUV;
      ovrp_v += overlay_pitchUV;
    }

  }

  dst_pitch *= sizeof(pixel_t);
  dst_pitchUV *= sizeof(pixel_t);
  overlay_pitch *= sizeof(pixel_t);
  overlay_pitchUV *= sizeof(pixel_t);
  mask_pitch *= sizeof(pixel_t);

  // make luma
  if constexpr (!lumaonly && maskMode != MASK444)
    layer_yuv_lighten_darken_c<mode, MASK444, pixel_t, lessthan16bits, true /* lumaonly*/, has_alpha>(
      dstp8, dstp8_u, dstp8_v, //dstp8_a,
      ovrp8, ovrp8_u, ovrp8_v, maskp8,
      dst_pitch, dst_pitchUV, overlay_pitch, overlay_pitchUV, mask_pitch,
      width, height, level, thresh, bits_per_pixel);
}

template<int mode, MaskMode maskMode, bool lumaonly, bool has_alpha>
static void layer_yuv_lighten_darken_f_c(
  BYTE* dstp8, BYTE* dstp8_u, BYTE* dstp8_v /*, BYTE* dstp8_a*/,
  const BYTE* ovrp8, const BYTE* ovrp8_u, const BYTE* ovrp8_v, const BYTE* maskp8,
  int dst_pitch, int dst_pitchUV,
  int overlay_pitch, int overlay_pitchUV,
  int mask_pitch,
  int width, int height, float opacity, float thresh) {

  float* dstp = reinterpret_cast<float*>(dstp8);
  float* dstp_u = reinterpret_cast<float*>(dstp8_u);
  float* dstp_v = reinterpret_cast<float*>(dstp8_v);
  //float* dstp_a = reinterpret_cast<float *>(dstp8_a);

  const float* ovrp = reinterpret_cast<const float*>(ovrp8);
  const float* ovrp_u = reinterpret_cast<const float*>(ovrp8_u);
  const float* ovrp_v = reinterpret_cast<const float*>(ovrp8_v);
  const float* maskp = reinterpret_cast<const float*>(maskp8);

  dst_pitch /= sizeof(float);
  dst_pitchUV /= sizeof(float);
  overlay_pitch /= sizeof(float);
  overlay_pitchUV /= sizeof(float);
  mask_pitch /= sizeof(float);

  const int cwidth = (maskMode == MASK444) ? width : (maskMode == MASK411) ? width >> 2 : width >> 1; // 444:/1  420,422:/2  411:/4
  const int cheight = (maskMode == MASK444 || maskMode == MASK422 || maskMode == MASK422_MPEG2 || maskMode == MASK422_TOPLEFT || maskMode == MASK411) ? height : height >> 1; // 444,422,411:/1  420:/2

  // In lighten/darken we need 3 buffers:
  std::vector<float> ovr_buffer;
  std::vector<float> src_buffer;
  std::vector<float> mask_buffer;

  // precalculate mask buffer et al.
  if constexpr (maskMode != MASK444) {
    ovr_buffer.resize(cwidth);
    src_buffer.resize(cwidth);
    if constexpr (has_alpha)
      mask_buffer.resize(cwidth);
  }

  // for subsampled color spaces first do chroma, because luma is used for decision
  // second pass will do luma only
  for (int y = 0; y < cheight; ++y) {
    const float* ovr_ptr = prepare_effective_mask_for_row_f<maskMode>(ovrp, overlay_pitch, cwidth, ovr_buffer);
    const float* src_ptr = prepare_effective_mask_for_row_f<maskMode>(dstp, dst_pitch, cwidth, src_buffer);
    const float* effective_mask_ptr = nullptr;
    if constexpr (has_alpha) {
      effective_mask_ptr = prepare_effective_mask_for_row_f<maskMode>(maskp, mask_pitch, cwidth, mask_buffer);
    }

    for (int x = 0; x < cwidth; ++x) {
      float ovr = ovr_ptr[x];
      float src = src_ptr[x];
      float effective_mask = has_alpha ? effective_mask_ptr[x] : 0;

      const float alpha = has_alpha ? effective_mask * opacity : opacity;

      float alpha_mask;
      if constexpr (mode == LIGHTEN)
        alpha_mask = ovr > (src + thresh) ? alpha : 0; // YUY2 was wrong: alpha_mask = (thresh + ovr) > src ? level : 0;
      else // DARKEN
        alpha_mask = ovr < (src - thresh) ? alpha : 0; // YUY2 was wrong: alpha_mask = (thresh + src) > ovr ? level : 0;

      if constexpr (!lumaonly)
      {
        // chroma u,v
        dstp_u[x] = dstp_u[x] + (ovrp_u[x] - dstp_u[x]) * alpha_mask;
        dstp_v[x] = dstp_v[x] + (ovrp_v[x] - dstp_v[x]) * alpha_mask;
        //dstp_a[x] = dstp_a[x] + (maskp[x] - dstp_a[x]) * alpha_mask;
      }

      // for 444: update here, width/height is the same as for chroma
      if constexpr (maskMode == MASK444)
        dstp[x] = dstp[x] + (ovrp[x] - dstp[x]) * alpha_mask;
    }
    if constexpr (maskMode == MASK420 || maskMode == MASK420_MPEG2 || maskMode == MASK420_TOPLEFT) {
      dstp += dst_pitch * 2; // skip vertical subsampling
      ovrp += overlay_pitch * 2;
      if constexpr (has_alpha) {
        //dstp_a += dst_pitch * 2;
        maskp += mask_pitch * 2;
      }
    }
    else {
      dstp += dst_pitch;
      ovrp += overlay_pitch;
      if constexpr (has_alpha) {
        //dstp_a += dst_pitch;
        maskp += mask_pitch;
      }
    }

    if constexpr (!lumaonly) {
      dstp_u += dst_pitchUV;
      dstp_v += dst_pitchUV;
      ovrp_u += overlay_pitchUV;
      ovrp_v += overlay_pitchUV;
    }
  }

  dst_pitch *= sizeof(float);
  dst_pitchUV *= sizeof(float);
  overlay_pitch *= sizeof(float);
  overlay_pitchUV *= sizeof(float);
  mask_pitch *= sizeof(float);

  // make luma
  if constexpr (!lumaonly && maskMode != MASK444)
    layer_yuv_lighten_darken_f_c<mode, MASK444, true /* lumaonly*/, has_alpha>(
      dstp8, dstp8_u, dstp8_v, //dstp8_a,
      ovrp8, ovrp8_u, ovrp8_v, maskp8,
      dst_pitch, dst_pitchUV, overlay_pitch, overlay_pitchUV, mask_pitch,
      width, height, opacity, thresh);
}

DISABLE_WARNING_POP

// Dispatchers

static void get_layer_yuv_lighten_darken_functions(bool isLighten, int placement, VideoInfo& vi, int bits_per_pixel, /*out*/layer_yuv_lighten_darken_c_t** layer_fn, /*out*/layer_yuv_lighten_darken_f_c_t** layer_f_fn) {

#define YUV_LIGHTEN_DARKEN_DISPATCH(L_or_D, MaskType, lumaonly, has_alpha) \
      { if (bits_per_pixel == 8) \
        *layer_fn = layer_yuv_lighten_darken_c<L_or_D, MaskType, uint8_t, true /*lessthan16bits*/, lumaonly /*lumaonly*/, has_alpha /*has_alpha*/>; \
      else if (bits_per_pixel < 16) \
        *layer_fn = layer_yuv_lighten_darken_c<L_or_D, MaskType, uint16_t, true/*lessthan16bits*/, lumaonly /*lumaonly*/, has_alpha /*has_alpha*/>; \
      else if (bits_per_pixel == 16) \
        *layer_fn = layer_yuv_lighten_darken_c<L_or_D, MaskType, uint16_t, false/*lessthan16bits*/, lumaonly /*lumaonly*/, has_alpha /*has_alpha*/>; \
      else /* float */ \
        *layer_f_fn = layer_yuv_lighten_darken_f_c<L_or_D, MaskType, lumaonly /*lumaonly*/, has_alpha /*has_alpha*/>; \
}

  if (isLighten) {

    if (vi.IsYV411())
      *layer_fn = layer_yuv_lighten_darken_c<LIGHTEN, MASK411, uint8_t, true /*lessthan16bits*/, false /*lumaonly*/, false /*has_alpha*/>;
    else if (vi.Is420())
    {
      if (placement == PLACEMENT_MPEG1)
        YUV_LIGHTEN_DARKEN_DISPATCH(LIGHTEN, MASK420, false, false)
      else if (placement == PLACEMENT_TOPLEFT)
        YUV_LIGHTEN_DARKEN_DISPATCH(LIGHTEN, MASK420_TOPLEFT, false, false)
      else
        YUV_LIGHTEN_DARKEN_DISPATCH(LIGHTEN, MASK420_MPEG2, false, false)
        // PLACEMENT_MPEG2
    }
    else if (vi.Is422())
    {
      if (placement == PLACEMENT_MPEG1)
        YUV_LIGHTEN_DARKEN_DISPATCH(LIGHTEN, MASK422, false, false)
      else if (placement == PLACEMENT_TOPLEFT)
        YUV_LIGHTEN_DARKEN_DISPATCH(LIGHTEN, MASK422_TOPLEFT, false, false)
      else
        YUV_LIGHTEN_DARKEN_DISPATCH(LIGHTEN, MASK422_MPEG2, false, false)
        // PLACEMENT_MPEG2
    }
    else if (vi.Is444())
      YUV_LIGHTEN_DARKEN_DISPATCH(LIGHTEN, MASK444, false, false)
    else if (vi.IsY())
      YUV_LIGHTEN_DARKEN_DISPATCH(LIGHTEN, MASK444, true, false)
  }
  else {
    // darken
    if (vi.IsYV411())
      *layer_fn = layer_yuv_lighten_darken_c<DARKEN, MASK411, uint8_t, true /*lessthan16bits*/, false /*lumaonly*/, false /*has_alpha*/>;
    else if (vi.Is420())
    {
      if (placement == PLACEMENT_MPEG1)
        YUV_LIGHTEN_DARKEN_DISPATCH(DARKEN, MASK420, false, false)
      else if (placement == PLACEMENT_TOPLEFT)
        YUV_LIGHTEN_DARKEN_DISPATCH(DARKEN, MASK420_TOPLEFT, false, false)
      else // PLACEMENT_MPEG2
        YUV_LIGHTEN_DARKEN_DISPATCH(DARKEN, MASK420_MPEG2, false, false)
    }
    else if (vi.Is422())
    {
      if (placement == PLACEMENT_MPEG1)
        YUV_LIGHTEN_DARKEN_DISPATCH(DARKEN, MASK422, false, false)
      else if (placement == PLACEMENT_TOPLEFT)
        YUV_LIGHTEN_DARKEN_DISPATCH(DARKEN, MASK422_TOPLEFT, false, false)
      else // PLACEMENT_MPEG2
        YUV_LIGHTEN_DARKEN_DISPATCH(DARKEN, MASK422_MPEG2, false, false)
    }
    else if (vi.Is444())
      YUV_LIGHTEN_DARKEN_DISPATCH(DARKEN, MASK444, false, false)
    else if (vi.IsY())
      YUV_LIGHTEN_DARKEN_DISPATCH(DARKEN, MASK444, true, false)
  }
#undef YUV_LIGHTEN_DARKEN_DISPATCH
}


static void get_layer_yuv_mul_functions(
  bool is_chroma, bool use_chroma, bool hasAlpha,
  int placement, VideoInfo& vi, int bits_per_pixel,
  /*out*/layer_yuv_mul_c_t** layer_fn,
  /*out*/layer_yuv_mul_f_c_t** layer_f_fn)
{
#define YUV_MUL_DISPATCH(MaskType, is_chroma, use_chroma, has_alpha) \
  { if (bits_per_pixel == 8) \
    *layer_fn = layer_yuv_mul_c<MaskType, uint8_t, true /*lessthan16bits*/, is_chroma, use_chroma, has_alpha>; \
  else if (bits_per_pixel < 16) \
    *layer_fn = layer_yuv_mul_c<MaskType, uint16_t, true /*lessthan16bits*/, is_chroma, use_chroma, has_alpha>; \
  else if (bits_per_pixel == 16) \
    *layer_fn = layer_yuv_mul_c<MaskType, uint16_t, false /*lessthan16bits*/, is_chroma, use_chroma, has_alpha>; \
  else /* float */ \
    *layer_f_fn = layer_yuv_mul_f_c<MaskType, is_chroma, use_chroma, has_alpha>; \
  }

  if (is_chroma) // not luma channel
  {
    if (vi.IsYV411())
    {
      if (use_chroma)
        *layer_fn = layer_yuv_mul_c<MASK411, uint8_t, true /*lessthan16bits*/, true, true, false>;
      else
        *layer_fn = layer_yuv_mul_c<MASK411, uint8_t, true /*lessthan16bits*/, true, false, false>;
    }
    else if (vi.Is420())
    {
      if (placement == PLACEMENT_MPEG1) {
        if (hasAlpha) {
          if (use_chroma) YUV_MUL_DISPATCH(MASK420, true, true, true)
          else YUV_MUL_DISPATCH(MASK420, true, false, true)
        }
        else {
          if (use_chroma) YUV_MUL_DISPATCH(MASK420, true, true, false)
          else YUV_MUL_DISPATCH(MASK420, true, false, false)
        }
      }
      else if (placement == PLACEMENT_TOPLEFT) {
        if (hasAlpha) {
          if (use_chroma) YUV_MUL_DISPATCH(MASK420_TOPLEFT, true, true, true)
          else YUV_MUL_DISPATCH(MASK420_TOPLEFT, true, false, true)
        }
        else {
          if (use_chroma) YUV_MUL_DISPATCH(MASK420_TOPLEFT, true, true, false)
          else YUV_MUL_DISPATCH(MASK420_TOPLEFT, true, false, false)
        }
      }
      else {
        if (hasAlpha) {
          if (use_chroma) YUV_MUL_DISPATCH(MASK420_MPEG2, true, true, true)
          else YUV_MUL_DISPATCH(MASK420_MPEG2, true, false, true)
        }
        else {
          if (use_chroma) YUV_MUL_DISPATCH(MASK420_MPEG2, true, true, false)
          else YUV_MUL_DISPATCH(MASK420_MPEG2, true, false, false)
        }
      }
    }
    else if (vi.Is422())
    {
      if (placement == PLACEMENT_MPEG1) {
        if (hasAlpha) {
          if (use_chroma) YUV_MUL_DISPATCH(MASK422, true, true, true)
          else YUV_MUL_DISPATCH(MASK422, true, false, true)
        }
        else {
          if (use_chroma) YUV_MUL_DISPATCH(MASK422, true, true, false)
          else YUV_MUL_DISPATCH(MASK422, true, false, false)
        }
      }
      else if (placement == PLACEMENT_TOPLEFT) {
        if (hasAlpha) {
          if (use_chroma) YUV_MUL_DISPATCH(MASK422_TOPLEFT, true, true, true)
          else YUV_MUL_DISPATCH(MASK422_TOPLEFT, true, false, true)
        }
        else {
          if (use_chroma) YUV_MUL_DISPATCH(MASK422_TOPLEFT, true, true, false)
          else YUV_MUL_DISPATCH(MASK422_TOPLEFT, true, false, false)
        }
      }
      else {
        if (hasAlpha) {
          if (use_chroma) YUV_MUL_DISPATCH(MASK422_MPEG2, true, true, true)
          else YUV_MUL_DISPATCH(MASK422_MPEG2, true, false, true)
        }
        else {
          if (use_chroma) YUV_MUL_DISPATCH(MASK422_MPEG2, true, true, false)
          else YUV_MUL_DISPATCH(MASK422_MPEG2, true, false, false)
        }
      }
    }
    else if (vi.Is444())
    {
      if (hasAlpha) {
        if (use_chroma) YUV_MUL_DISPATCH(MASK444, true, true, true)
        else YUV_MUL_DISPATCH(MASK444, true, false, true)
      }
      else {
        if (use_chroma) YUV_MUL_DISPATCH(MASK444, true, true, false)
        else YUV_MUL_DISPATCH(MASK444, true, false, false)
      }
    }
  }
  else // luma channel
  {
    if (hasAlpha)
      YUV_MUL_DISPATCH(MASK444, false, false, true)
    else
      YUV_MUL_DISPATCH(MASK444, false, false, false)
  }
#undef YUV_MUL_DISPATCH
}

static void get_layer_yuv_add_functions(
  bool is_chroma, bool use_chroma, bool hasAlpha,
  int placement, VideoInfo& vi, int bits_per_pixel,
  /*out*/layer_yuv_add_c_t** layer_fn,
  /*out*/layer_yuv_add_f_c_t** layer_f_fn)
{
#define YUV_ADD_SUBTRACT_DISPATCH(MaskType, is_chroma, use_chroma, has_alpha) \
  { if (bits_per_pixel == 8) \
    *layer_fn = layer_yuv_add_c<MaskType, uint8_t, true /*lessthan16bits*/, is_chroma, use_chroma, has_alpha>; \
  else if (bits_per_pixel < 16) \
    *layer_fn = layer_yuv_add_c<MaskType, uint16_t, true /*lessthan16bits*/, is_chroma, use_chroma, has_alpha>; \
  else if (bits_per_pixel == 16) \
    *layer_fn = layer_yuv_add_c<MaskType, uint16_t, false /*lessthan16bits*/, is_chroma, use_chroma, has_alpha>; \
  else /* float */ \
    *layer_f_fn = layer_yuv_add_f_c<MaskType, is_chroma, use_chroma, has_alpha>; \
  }

  if (is_chroma) // not luma channel
  {
    if (vi.IsYV411())
    {
      if (use_chroma)
        *layer_fn = layer_yuv_add_c<MASK411, uint8_t, true /*lessthan16bits*/, true, true, false>;
      else
        *layer_fn = layer_yuv_add_c<MASK411, uint8_t, true /*lessthan16bits*/, true, false, false>;
    }
    else if (vi.Is420())
    {
      if (placement == PLACEMENT_MPEG1) {
        if (hasAlpha) {
          if (use_chroma) YUV_ADD_SUBTRACT_DISPATCH(MASK420, true, true, true)
          else YUV_ADD_SUBTRACT_DISPATCH(MASK420, true, false, true)
        }
        else {
          if (use_chroma) YUV_ADD_SUBTRACT_DISPATCH(MASK420, true, true, false)
          else YUV_ADD_SUBTRACT_DISPATCH(MASK420, true, false, false)
        }
      }
      else if (placement == PLACEMENT_TOPLEFT) {
        if (hasAlpha) {
          if (use_chroma) YUV_ADD_SUBTRACT_DISPATCH(MASK420_TOPLEFT, true, true, true)
          else YUV_ADD_SUBTRACT_DISPATCH(MASK420_TOPLEFT, true, false, true)
        }
        else {
          if (use_chroma) YUV_ADD_SUBTRACT_DISPATCH(MASK420_TOPLEFT, true, true, false)
          else YUV_ADD_SUBTRACT_DISPATCH(MASK420_TOPLEFT, true, false, false)
        }
      }
      else {
        if (hasAlpha) {
          if (use_chroma) YUV_ADD_SUBTRACT_DISPATCH(MASK420_MPEG2, true, true, true)
          else YUV_ADD_SUBTRACT_DISPATCH(MASK420_MPEG2, true, false, true)
        }
        else {
          if (use_chroma) YUV_ADD_SUBTRACT_DISPATCH(MASK420_MPEG2, true, true, false)
          else YUV_ADD_SUBTRACT_DISPATCH(MASK420_MPEG2, true, false, false)
        }
      }
    }
    else if (vi.Is422())
    {
      if (placement == PLACEMENT_MPEG1) {
        if (hasAlpha) {
          if (use_chroma) YUV_ADD_SUBTRACT_DISPATCH(MASK422, true, true, true)
          else YUV_ADD_SUBTRACT_DISPATCH(MASK422, true, false, true)
        }
        else {
          if (use_chroma) YUV_ADD_SUBTRACT_DISPATCH(MASK422, true, true, false)
          else YUV_ADD_SUBTRACT_DISPATCH(MASK422, true, false, false)
        }
      }
      else if (placement == PLACEMENT_TOPLEFT) {
        if (hasAlpha) {
          if (use_chroma) YUV_ADD_SUBTRACT_DISPATCH(MASK422_TOPLEFT, true, true, true)
          else YUV_ADD_SUBTRACT_DISPATCH(MASK422_TOPLEFT, true, false, true)
        }
        else {
          if (use_chroma) YUV_ADD_SUBTRACT_DISPATCH(MASK422_TOPLEFT, true, true, false)
          else YUV_ADD_SUBTRACT_DISPATCH(MASK422_TOPLEFT, true, false, false)
        }
      }
      else {
        if (hasAlpha) {
          if (use_chroma) YUV_ADD_SUBTRACT_DISPATCH(MASK422_MPEG2, true, true, true)
          else YUV_ADD_SUBTRACT_DISPATCH(MASK422_MPEG2, true, false, true)
        }
        else {
          if (use_chroma) YUV_ADD_SUBTRACT_DISPATCH(MASK422_MPEG2, true, true, false)
          else YUV_ADD_SUBTRACT_DISPATCH(MASK422_MPEG2, true, false, false)
        }
      }
    }
    else if (vi.Is444())
    {
      if (hasAlpha) {
        if (use_chroma) YUV_ADD_SUBTRACT_DISPATCH(MASK444, true, true, true)
        else YUV_ADD_SUBTRACT_DISPATCH(MASK444, true, false, true)
      }
      else {
        if (use_chroma) YUV_ADD_SUBTRACT_DISPATCH(MASK444, true, true, false)
        else YUV_ADD_SUBTRACT_DISPATCH(MASK444, true, false, false)
      }
    }
  }
  else // luma channel
  {
    if (hasAlpha)
      YUV_ADD_SUBTRACT_DISPATCH(MASK444, false, false, true)
    else
      YUV_ADD_SUBTRACT_DISPATCH(MASK444, false, false, false)
  }
#undef YUV_ADD_SUBTRACT_DISPATCH
}

/* planar rgb */

template<int mode, typename pixel_t, bool lessthan16bits, bool has_alpha, bool blend_alpha>
static void layer_planarrgb_lighten_darken_c(BYTE** dstp8, const BYTE** ovrp8, const BYTE* maskp8, int dst_pitch, int overlay_pitch, int mask_pitch, int width, int height, int level, int thresh, int bits_per_pixel) {
  pixel_t* dstp_g = reinterpret_cast<pixel_t*>(dstp8[0]);
  pixel_t* dstp_b = reinterpret_cast<pixel_t*>(dstp8[1]);
  pixel_t* dstp_r = reinterpret_cast<pixel_t*>(dstp8[2]);
  // dstp8[3]: written only when blend_alpha=true (both clips have alpha).
  pixel_t* dstp_a;
  if constexpr (blend_alpha)
    dstp_a = reinterpret_cast<pixel_t*>(dstp8[3]);
  const pixel_t* ovrp_g = reinterpret_cast<const pixel_t*>(ovrp8[0]);
  const pixel_t* ovrp_b = reinterpret_cast<const pixel_t*>(ovrp8[1]);
  const pixel_t* ovrp_r = reinterpret_cast<const pixel_t*>(ovrp8[2]);
  // maskp: per-pixel blend weight — decoupled from ovrp8[3] so a future mask clip can be wired in.
  const pixel_t* maskp = reinterpret_cast<const pixel_t*>(maskp8);
  // alpha_target: value blended into dstp_a (only used when blend_alpha=true).
  // For plain Add this is the same plane as maskp; for Subtract it is the inverted A plane.
  const pixel_t* alpha_target;
  if constexpr (blend_alpha)
    alpha_target = reinterpret_cast<const pixel_t*>(ovrp8[3]);

  dst_pitch /= sizeof(pixel_t);
  overlay_pitch /= sizeof(pixel_t);
  mask_pitch /= sizeof(pixel_t);

  using calc_t = typename std::conditional < lessthan16bits, int, int64_t>::type; // for non-overflowing 16 bit alpha_mul
  if constexpr (sizeof(pixel_t) == 1)
    bits_per_pixel = 8; // make quasi constexpr
  else if constexpr (sizeof(pixel_t) == 2 && !lessthan16bits)
    bits_per_pixel = 16; // make quasi constexpr

  const int rounder = 1 << (bits_per_pixel - 1);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      calc_t alpha = has_alpha ? ((calc_t)maskp[x] * level + 1) >> bits_per_pixel : level;

      calc_t luma_ovr = (cyb * ovrp_b[x] + cyg * ovrp_g[x] + cyr * ovrp_r[x]) >> 15; // no rounding, not really needed here
      calc_t luma_src = (cyb * dstp_b[x] + cyg * dstp_g[x] + cyr * dstp_r[x]) >> 15;

      if constexpr (mode == LIGHTEN)
        alpha = luma_ovr > luma_src + thresh ? alpha : 0;
      else // DARKEN
        alpha = luma_ovr < luma_src - thresh ? alpha : 0;

      dstp_r[x] = (pixel_t)(dstp_r[x] + (((ovrp_r[x] - dstp_r[x]) * alpha + rounder) >> bits_per_pixel));
      dstp_g[x] = (pixel_t)(dstp_g[x] + (((ovrp_g[x] - dstp_g[x]) * alpha + rounder) >> bits_per_pixel));
      dstp_b[x] = (pixel_t)(dstp_b[x] + (((ovrp_b[x] - dstp_b[x]) * alpha + rounder) >> bits_per_pixel));
      // alpha channel: same Add formula; alpha_target may differ from maskp (Subtract case).
      if constexpr (blend_alpha)
        dstp_a[x] = (pixel_t)(dstp_a[x] + (((alpha_target[x] - dstp_a[x]) * alpha + rounder) >> bits_per_pixel));
    }
    dstp_g += dst_pitch;
    dstp_b += dst_pitch;
    dstp_r += dst_pitch;
    if constexpr (blend_alpha)
      dstp_a += dst_pitch;
    ovrp_g += overlay_pitch;
    ovrp_b += overlay_pitch;
    ovrp_r += overlay_pitch;
    if constexpr (has_alpha)
      maskp += mask_pitch;
    if constexpr (blend_alpha)
      alpha_target += overlay_pitch;
  }
}

template<int mode, bool has_alpha, bool blend_alpha>
static void layer_planarrgb_lighten_darken_f_c(BYTE** dstp8, const BYTE** ovrp8, const BYTE* maskp8, int dst_pitch, int overlay_pitch, int mask_pitch, int width, int height, float opacity, float thresh) {
  float* dstp_g = reinterpret_cast<float*>(dstp8[0]);
  float* dstp_b = reinterpret_cast<float*>(dstp8[1]);
  float* dstp_r = reinterpret_cast<float*>(dstp8[2]);
  // dstp8[3]: written only when blend_alpha=true (both clips have alpha).
  float* dstp_a;
  if constexpr (blend_alpha)
    dstp_a = reinterpret_cast<float*>(dstp8[3]);
  const float* ovrp_g = reinterpret_cast<const float*>(ovrp8[0]);
  const float* ovrp_b = reinterpret_cast<const float*>(ovrp8[1]);
  const float* ovrp_r = reinterpret_cast<const float*>(ovrp8[2]);
  // maskp: per-pixel blend weight — decoupled from ovrp8[3].
  const float* maskp = reinterpret_cast<const float*>(maskp8);
  const float* alpha_target;
  if constexpr (blend_alpha)
    alpha_target = reinterpret_cast<const float*>(ovrp8[3]);

  dst_pitch /= sizeof(float);
  overlay_pitch /= sizeof(float);
  mask_pitch /= sizeof(float);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float alpha = has_alpha ? maskp[x] * opacity : opacity;

      float luma_ovr = cyb_f * ovrp_b[x] + cyg_f * ovrp_g[x] + cyr_f * ovrp_r[x];
      float luma_src = cyb_f * dstp_b[x] + cyg_f * dstp_g[x] + cyr_f * dstp_r[x];

      if constexpr (mode == LIGHTEN)
        alpha = luma_ovr > luma_src + thresh ? alpha : 0;
      else // DARKEN
        alpha = luma_ovr < luma_src - thresh ? alpha : 0;

      dstp_r[x] = dstp_r[x] + (ovrp_r[x] - dstp_r[x]) * alpha;
      dstp_g[x] = dstp_g[x] + (ovrp_g[x] - dstp_g[x]) * alpha;
      dstp_b[x] = dstp_b[x] + (ovrp_b[x] - dstp_b[x]) * alpha;
      if constexpr (blend_alpha)
        dstp_a[x] = dstp_a[x] + (alpha_target[x] - dstp_a[x]) * alpha;
    }
    dstp_g += dst_pitch;
    dstp_b += dst_pitch;
    dstp_r += dst_pitch;
    if constexpr (blend_alpha)
      dstp_a += dst_pitch;
    ovrp_g += overlay_pitch;
    ovrp_b += overlay_pitch;
    ovrp_r += overlay_pitch;
    if constexpr (has_alpha)
      maskp += mask_pitch;
    if constexpr (blend_alpha)
      alpha_target += overlay_pitch;
  }
}


static void get_layer_planarrgb_lighten_darken_functions(bool isLighten, bool hasAlpha, bool blendAlpha, int bits_per_pixel, /*out*/layer_planarrgb_lighten_darken_c_t** layer_fn, /*out*/layer_planarrgb_lighten_darken_f_c_t** layer_f_fn) {

#define PLANARRGB_LD_DISPATCH(LorD, has_alpha, blend_alpha) \
      { if (bits_per_pixel == 8) \
        *layer_fn = layer_planarrgb_lighten_darken_c<LorD, uint8_t, true /*lessthan16bits*/, has_alpha, blend_alpha>; \
      else if (bits_per_pixel < 16) \
        *layer_fn = layer_planarrgb_lighten_darken_c<LorD, uint16_t, true /*lessthan16bits*/, has_alpha, blend_alpha>; \
      else if (bits_per_pixel == 16) \
        *layer_fn = layer_planarrgb_lighten_darken_c<LorD, uint16_t, false /*lessthan16bits*/, has_alpha, blend_alpha>; \
      else /* float */ \
        *layer_f_fn = layer_planarrgb_lighten_darken_f_c<LorD, has_alpha, blend_alpha>; \
      }

  if (isLighten) {
    if (hasAlpha) {
      if (blendAlpha) { PLANARRGB_LD_DISPATCH(LIGHTEN, true, true)  }
      else            { PLANARRGB_LD_DISPATCH(LIGHTEN, true, false) }
    }
    else {
      PLANARRGB_LD_DISPATCH(LIGHTEN, false, false)
    }
  } // lighten end
  else {
    if (hasAlpha) {
      if (blendAlpha) { PLANARRGB_LD_DISPATCH(DARKEN, true, true)  }
      else            { PLANARRGB_LD_DISPATCH(DARKEN, true, false) }
    }
    else {
      PLANARRGB_LD_DISPATCH(DARKEN, false, false)
    }
  }
#undef PLANARRGB_LD_DISPATCH
}

// subtract=false only: overlay clip is pre-inverted by Layer::Create when op="Subtract".
template<typename pixel_t, bool lessthan16bits, bool chroma, bool has_alpha, bool blend_alpha>
static void layer_planarrgb_add_c(BYTE** dstp8, const BYTE** ovrp8, const BYTE* maskp8, int dst_pitch, int overlay_pitch, int mask_pitch, int width, int height, int level, int bits_per_pixel) {
  pixel_t* dstp_g = reinterpret_cast<pixel_t*>(dstp8[0]);
  pixel_t* dstp_b = reinterpret_cast<pixel_t*>(dstp8[1]);
  pixel_t* dstp_r = reinterpret_cast<pixel_t*>(dstp8[2]);
  // dstp8[3]: written only when blend_alpha=true (both clips have alpha).
  pixel_t* dstp_a;
  if constexpr (blend_alpha)
    dstp_a = reinterpret_cast<pixel_t*>(dstp8[3]);
  const pixel_t* ovrp_g = reinterpret_cast<const pixel_t*>(ovrp8[0]);
  const pixel_t* ovrp_b = reinterpret_cast<const pixel_t*>(ovrp8[1]);
  const pixel_t* ovrp_r = reinterpret_cast<const pixel_t*>(ovrp8[2]);
  // maskp: per-pixel blend weight — decoupled from ovrp8[3] to support future mask clip param.
  const pixel_t* maskp = reinterpret_cast<const pixel_t*>(maskp8);
  // alpha_target: the value blended into dstp_a.
  // For plain Add: same as maskp (original overlay A).
  // For Subtract: ovrp8[3] is the inverted A; maskp is the saved pre-invert A.
  const pixel_t* alpha_target;
  if constexpr (blend_alpha)
    alpha_target = reinterpret_cast<const pixel_t*>(ovrp8[3]);

  dst_pitch /= sizeof(pixel_t);
  overlay_pitch /= sizeof(pixel_t);
  mask_pitch /= sizeof(pixel_t);

  using calc_t = typename std::conditional < lessthan16bits, int, int64_t>::type; // for non-overflowing 16 bit alpha_mul
  if constexpr (sizeof(pixel_t) == 1)
    bits_per_pixel = 8; // make quasi constexpr
  else if constexpr (sizeof(pixel_t) == 2 && !lessthan16bits)
    bits_per_pixel = 16; // make quasi constexpr

  const int rounder = 1 << (bits_per_pixel - 1);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      calc_t alpha = has_alpha ? ((calc_t)maskp[x] * level + 1) >> bits_per_pixel : level;
      if constexpr (chroma) {
        dstp_r[x] = (pixel_t)(dstp_r[x] + (((ovrp_r[x] - dstp_r[x]) * alpha + rounder) >> bits_per_pixel));
        dstp_g[x] = (pixel_t)(dstp_g[x] + (((ovrp_g[x] - dstp_g[x]) * alpha + rounder) >> bits_per_pixel));
        dstp_b[x] = (pixel_t)(dstp_b[x] + (((ovrp_b[x] - dstp_b[x]) * alpha + rounder) >> bits_per_pixel));
      }
      else { // use luma instead of overlay
        calc_t luma = (cyb * ovrp_b[x] + cyg * ovrp_g[x] + cyr * ovrp_r[x]) >> 15; // no rounding not really needed here

        dstp_r[x] = (pixel_t)(dstp_r[x] + (((luma - dstp_r[x]) * alpha + rounder) >> bits_per_pixel));
        dstp_g[x] = (pixel_t)(dstp_g[x] + (((luma - dstp_g[x]) * alpha + rounder) >> bits_per_pixel));
        dstp_b[x] = (pixel_t)(dstp_b[x] + (((luma - dstp_b[x]) * alpha + rounder) >> bits_per_pixel));
      }
      // alpha channel: Add formula; alpha_target may differ from maskp (Subtract case).
      if constexpr (blend_alpha)
        dstp_a[x] = (pixel_t)(dstp_a[x] + (((alpha_target[x] - dstp_a[x]) * alpha + rounder) >> bits_per_pixel));
    }
    dstp_g += dst_pitch;
    dstp_b += dst_pitch;
    dstp_r += dst_pitch;
    if constexpr (blend_alpha)
      dstp_a += dst_pitch;
    ovrp_g += overlay_pitch;
    ovrp_b += overlay_pitch;
    ovrp_r += overlay_pitch;
    if constexpr (has_alpha)
      maskp += mask_pitch;
    if constexpr (blend_alpha)
      alpha_target += overlay_pitch;
  }
}

// subtract=false only: overlay clip is pre-inverted by Layer::Create when op="Subtract".
template<bool chroma, bool has_alpha, bool blend_alpha>
static void layer_planarrgb_add_f_c(BYTE** dstp8, const BYTE** ovrp8, const BYTE* maskp8, int dst_pitch, int overlay_pitch, int mask_pitch, int width, int height, float opacity) {
  float* dstp_g = reinterpret_cast<float*>(dstp8[0]);
  float* dstp_b = reinterpret_cast<float*>(dstp8[1]);
  float* dstp_r = reinterpret_cast<float*>(dstp8[2]);
  // dstp8[3]: written only when blend_alpha=true (both clips have alpha).
  float* dstp_a;
  if constexpr (blend_alpha)
    dstp_a = reinterpret_cast<float*>(dstp8[3]);
  const float* ovrp_g = reinterpret_cast<const float*>(ovrp8[0]);
  const float* ovrp_b = reinterpret_cast<const float*>(ovrp8[1]);
  const float* ovrp_r = reinterpret_cast<const float*>(ovrp8[2]);
  // maskp: per-pixel blend weight — decoupled from ovrp8[3].
  const float* maskp = reinterpret_cast<const float*>(maskp8);
  const float* alpha_target;
  if constexpr (blend_alpha)
    alpha_target = reinterpret_cast<const float*>(ovrp8[3]);

  dst_pitch /= sizeof(float);
  overlay_pitch /= sizeof(float);
  mask_pitch /= sizeof(float);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float alpha = has_alpha ? maskp[x] * opacity : opacity;

      if constexpr (chroma) {
        dstp_r[x] = dstp_r[x] + (ovrp_r[x] - dstp_r[x]) * alpha;
        dstp_g[x] = dstp_g[x] + (ovrp_g[x] - dstp_g[x]) * alpha;
        dstp_b[x] = dstp_b[x] + (ovrp_b[x] - dstp_b[x]) * alpha;
      }
      else { // use luma instead of overlay
        float luma = cyb_f * ovrp_b[x] + cyg_f * ovrp_g[x] + cyr_f * ovrp_r[x];
        dstp_r[x] = dstp_r[x] + (luma - dstp_r[x]) * alpha;
        dstp_g[x] = dstp_g[x] + (luma - dstp_g[x]) * alpha;
        dstp_b[x] = dstp_b[x] + (luma - dstp_b[x]) * alpha;
      }
      if constexpr (blend_alpha)
        dstp_a[x] = dstp_a[x] + (alpha_target[x] - dstp_a[x]) * alpha;
    }
    dstp_g += dst_pitch;
    dstp_b += dst_pitch;
    dstp_r += dst_pitch;
    if constexpr (blend_alpha)
      dstp_a += dst_pitch;
    ovrp_g += overlay_pitch;
    ovrp_b += overlay_pitch;
    ovrp_r += overlay_pitch;
    if constexpr (has_alpha)
      maskp += mask_pitch;
    if constexpr (blend_alpha)
      alpha_target += overlay_pitch;
  }
}

static void get_layer_planarrgb_add_functions(
  bool chroma, bool hasAlpha, bool blendAlpha, int bits_per_pixel,
  /*out*/layer_planarrgb_add_c_t** layer_fn,
  /*out*/layer_planarrgb_add_f_c_t** layer_f_fn)
{
#define PLANARRGB_ADD_DISPATCH(chroma, has_alpha, blend_alpha) \
  { if (bits_per_pixel == 8) \
    *layer_fn = layer_planarrgb_add_c<uint8_t, true /*lessthan16bits*/, chroma, has_alpha, blend_alpha>; \
  else if (bits_per_pixel < 16) \
    *layer_fn = layer_planarrgb_add_c<uint16_t, true /*lessthan16bits*/, chroma, has_alpha, blend_alpha>; \
  else if (bits_per_pixel == 16) \
    *layer_fn = layer_planarrgb_add_c<uint16_t, false /*lessthan16bits*/, chroma, has_alpha, blend_alpha>; \
  else /* float */ \
    *layer_f_fn = layer_planarrgb_add_f_c<chroma, has_alpha, blend_alpha>; \
  }

  if (hasAlpha) {
    if (chroma) {
      if (blendAlpha) { PLANARRGB_ADD_DISPATCH(true, true, true)  }
      else            { PLANARRGB_ADD_DISPATCH(true, true, false) }
    }
    else {
      if (blendAlpha) { PLANARRGB_ADD_DISPATCH(false, true, true)  }
      else            { PLANARRGB_ADD_DISPATCH(false, true, false) }
    }
  }
  else {
    if (chroma) {
      PLANARRGB_ADD_DISPATCH(true, false, false)
    }
    else {
      PLANARRGB_ADD_DISPATCH(false, false, false)
    }
  }
#undef PLANARRGB_ADD_DISPATCH
}

template<typename pixel_t, bool lessthan16bits, bool chroma, bool has_alpha, bool blend_alpha>
static void layer_planarrgb_mul_c(BYTE** dstp8, const BYTE** ovrp8, const BYTE* maskp8, int dst_pitch, int overlay_pitch, int mask_pitch, int width, int height, int level, int bits_per_pixel) {
  pixel_t* dstp_g = reinterpret_cast<pixel_t*>(dstp8[0]);
  pixel_t* dstp_b = reinterpret_cast<pixel_t*>(dstp8[1]);
  pixel_t* dstp_r = reinterpret_cast<pixel_t*>(dstp8[2]);
  // dstp8[3]: written only when blend_alpha=true (both clips have alpha).
  pixel_t* dstp_a;
  if constexpr (blend_alpha)
    dstp_a = reinterpret_cast<pixel_t*>(dstp8[3]);
  const pixel_t* ovrp_g = reinterpret_cast<const pixel_t*>(ovrp8[0]);
  const pixel_t* ovrp_b = reinterpret_cast<const pixel_t*>(ovrp8[1]);
  const pixel_t* ovrp_r = reinterpret_cast<const pixel_t*>(ovrp8[2]);
  // maskp: per-pixel blend weight — decoupled from ovrp8[3].
  const pixel_t* maskp = reinterpret_cast<const pixel_t*>(maskp8);
  // alpha_target: the value multiplied into dstp_a (only when blend_alpha=true).
  const pixel_t* alpha_target;
  if constexpr (blend_alpha)
    alpha_target = reinterpret_cast<const pixel_t*>(ovrp8[3]);

  dst_pitch /= sizeof(pixel_t);
  overlay_pitch /= sizeof(pixel_t);
  mask_pitch /= sizeof(pixel_t);

  using calc_t = typename std::conditional < lessthan16bits, int, int64_t>::type;
  if constexpr (sizeof(pixel_t) == 1)
    bits_per_pixel = 8; // make quasi constexpr
  else if constexpr (sizeof(pixel_t) == 2 && !lessthan16bits)
    bits_per_pixel = 16; // make quasi constexpr

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      calc_t alpha = has_alpha ? ((calc_t)maskp[x] * level + 1) >> bits_per_pixel : level;

      if constexpr (chroma) {
        dstp_r[x] = (pixel_t)(dstp_r[x] + ((((((calc_t)ovrp_r[x] * dstp_r[x]) >> bits_per_pixel) - dstp_r[x]) * alpha) >> bits_per_pixel));
        dstp_g[x] = (pixel_t)(dstp_g[x] + ((((((calc_t)ovrp_g[x] * dstp_g[x]) >> bits_per_pixel) - dstp_g[x]) * alpha) >> bits_per_pixel));
        dstp_b[x] = (pixel_t)(dstp_b[x] + ((((((calc_t)ovrp_b[x] * dstp_b[x]) >> bits_per_pixel) - dstp_b[x]) * alpha) >> bits_per_pixel));
      }
      else { // use luma instead of overlay
        calc_t luma = (cyb * ovrp_b[x] + cyg * ovrp_g[x] + cyr * ovrp_r[x]) >> 15; // no rounding not really needed here

        dstp_r[x] = (pixel_t)(dstp_r[x] + (((((luma * dstp_r[x]) >> bits_per_pixel) - dstp_r[x]) * alpha) >> bits_per_pixel));
        dstp_g[x] = (pixel_t)(dstp_g[x] + (((((luma * dstp_g[x]) >> bits_per_pixel) - dstp_g[x]) * alpha) >> bits_per_pixel));
        dstp_b[x] = (pixel_t)(dstp_b[x] + (((((luma * dstp_b[x]) >> bits_per_pixel) - dstp_b[x]) * alpha) >> bits_per_pixel));
      }
      // alpha channel: Mul formula; alpha_target provides the ovr_A value.
      if constexpr (blend_alpha)
        dstp_a[x] = (pixel_t)(dstp_a[x] + ((((((calc_t)alpha_target[x] * dstp_a[x]) >> bits_per_pixel) - dstp_a[x]) * alpha) >> bits_per_pixel));
    }
    dstp_g += dst_pitch;
    dstp_b += dst_pitch;
    dstp_r += dst_pitch;
    if constexpr (blend_alpha)
      dstp_a += dst_pitch;
    ovrp_g += overlay_pitch;
    ovrp_b += overlay_pitch;
    ovrp_r += overlay_pitch;
    if constexpr (has_alpha)
      maskp += mask_pitch;
    if constexpr (blend_alpha)
      alpha_target += overlay_pitch;
  }
}

template<bool chroma, bool has_alpha, bool blend_alpha>
static void layer_planarrgb_mul_f_c(BYTE** dstp8, const BYTE** ovrp8, const BYTE* maskp8, int dst_pitch, int overlay_pitch, int mask_pitch, int width, int height, float opacity) {
  float* dstp_g = reinterpret_cast<float*>(dstp8[0]);
  float* dstp_b = reinterpret_cast<float*>(dstp8[1]);
  float* dstp_r = reinterpret_cast<float*>(dstp8[2]);
  // dstp8[3]: written only when blend_alpha=true (both clips have alpha).
  float* dstp_a;
  if constexpr (blend_alpha)
    dstp_a = reinterpret_cast<float*>(dstp8[3]);
  const float* ovrp_g = reinterpret_cast<const float*>(ovrp8[0]);
  const float* ovrp_b = reinterpret_cast<const float*>(ovrp8[1]);
  const float* ovrp_r = reinterpret_cast<const float*>(ovrp8[2]);
  // maskp: per-pixel blend weight — decoupled from ovrp8[3].
  const float* maskp = reinterpret_cast<const float*>(maskp8);
  const float* alpha_target;
  if constexpr (blend_alpha)
    alpha_target = reinterpret_cast<const float*>(ovrp8[3]);

  dst_pitch /= sizeof(float);
  overlay_pitch /= sizeof(float);
  mask_pitch /= sizeof(float);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float alpha = has_alpha ? maskp[x] * opacity : opacity;

      if constexpr (chroma) {
        dstp_r[x] = dstp_r[x] + (ovrp_r[x] * dstp_r[x] - dstp_r[x]) * alpha;
        dstp_g[x] = dstp_g[x] + (ovrp_g[x] * dstp_g[x] - dstp_g[x]) * alpha;
        dstp_b[x] = dstp_b[x] + (ovrp_b[x] * dstp_b[x] - dstp_b[x]) * alpha;
      }
      else { // use luma instead of overlay
        float luma = cyb_f * ovrp_b[x] + cyg_f * ovrp_g[x] + cyr_f * ovrp_r[x];
        dstp_r[x] = dstp_r[x] + (luma * dstp_r[x] - dstp_r[x]) * alpha;
        dstp_g[x] = dstp_g[x] + (luma * dstp_g[x] - dstp_g[x]) * alpha;
        dstp_b[x] = dstp_b[x] + (luma * dstp_b[x] - dstp_b[x]) * alpha;
      }
      if constexpr (blend_alpha)
        dstp_a[x] = dstp_a[x] + (alpha_target[x] * dstp_a[x] - dstp_a[x]) * alpha;
    }
    dstp_g += dst_pitch;
    dstp_b += dst_pitch;
    dstp_r += dst_pitch;
    if constexpr (blend_alpha)
      dstp_a += dst_pitch;
    ovrp_g += overlay_pitch;
    ovrp_b += overlay_pitch;
    ovrp_r += overlay_pitch;
    if constexpr (has_alpha)
      maskp += mask_pitch;
    if constexpr (blend_alpha)
      alpha_target += overlay_pitch;
  }
}

static void get_layer_planarrgb_mul_functions(
  bool chroma, bool hasAlpha, bool blendAlpha, int bits_per_pixel,
  /*out*/layer_planarrgb_mul_c_t** layer_fn,
  /*out*/layer_planarrgb_mul_f_c_t** layer_f_fn)
{
#define PLANARRGB_MUL_DISPATCH(chroma, has_alpha, blend_alpha) \
  { if (bits_per_pixel == 8) \
    *layer_fn = layer_planarrgb_mul_c<uint8_t, true /*lessthan16bits*/, chroma, has_alpha, blend_alpha>; \
  else if (bits_per_pixel < 16) \
    *layer_fn = layer_planarrgb_mul_c<uint16_t, true /*lessthan16bits*/, chroma, has_alpha, blend_alpha>; \
  else if (bits_per_pixel == 16) \
    *layer_fn = layer_planarrgb_mul_c<uint16_t, false /*lessthan16bits*/, chroma, has_alpha, blend_alpha>; \
  else /* float */ \
    *layer_f_fn = layer_planarrgb_mul_f_c<chroma, has_alpha, blend_alpha>; \
  }

  if (hasAlpha) {
    if (chroma) {
      if (blendAlpha) { PLANARRGB_MUL_DISPATCH(true, true, true)  }
      else            { PLANARRGB_MUL_DISPATCH(true, true, false) }
    }
    else {
      if (blendAlpha) { PLANARRGB_MUL_DISPATCH(false, true, true)  }
      else            { PLANARRGB_MUL_DISPATCH(false, true, false) }
    }
  }
  else {
    if (chroma) {
      PLANARRGB_MUL_DISPATCH(true, false, false)
    }
    else {
      PLANARRGB_MUL_DISPATCH(false, false, false)
    }
  }
#undef PLANARRGB_MUL_DISPATCH
}

// ---------------------------------------------------------------------------
// Packed RGBA (RGB32 / RGB64) blend dispatcher — C reference.
// Returns the appropriate masked_blend_packedrgba_c instantiation for the
// given bit depth.  Subtract is handled by pre-inverting the overlay and
// passing a separate maskp8 pointer in the caller (Layer::Create/GetFrame).
// The C reference function selects the alpha source at runtime via the
// maskp8 null-check, so a single function covers both Add and Subtract.
// ---------------------------------------------------------------------------
static void get_layer_packedrgb_blend_functions(
  int bits_per_pixel,
  layer_packedrgb_blend_c_t** fn)
{
  if (bits_per_pixel == 8)
    *fn = masked_blend_packedrgba_c<uint8_t>;
  else  // 16-bit (RGB64)
    *fn = masked_blend_packedrgba_c<uint16_t>;
}

// Clean up rowprep macros (defined near top of this file or by including TU).
#undef LAYER_ROWPREP_FN
#undef LAYER_ROWPREP_LEVEL_FN

