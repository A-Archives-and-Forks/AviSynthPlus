// AviSynth+.  Copyright 2026- AviSynth+ Project
// https://avs-plus.net
// http://avisynth.nl
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA, or visit
// http://www.gnu.org/copyleft/gpl.html .
//
// Linking Avisynth statically or dynamically with other modules is making a
// combined work based on Avisynth.  Thus, the terms and conditions of the GNU
// General Public License cover the whole combination.
//
// As a special exception, the copyright holders of Avisynth give you
// permission to link Avisynth with independent modules that communicate with
// Avisynth solely through the interfaces defined in avisynth.h, regardless of the license
// terms of these independent modules, and to copy and distribute the
// resulting combined work under terms of your choice, provided that
// every copy of the combined work is accompanied by a complete copy of
// the source code of Avisynth (the version of Avisynth used to produce the
// combined work), being distributed under the terms of the GNU General
// Public License plus this exception.  An independent module is a module
// which is not derived from or based on Avisynth, such as 3rd-party filters,
// import and export plugins, or graphical user interfaces.

#include <avisynth.h>
#include "../blend_common.h"
#include "blend_common_neon.h"
#include <arm_neon.h>
#include <cstdint>
#include <type_traits>
#include <vector>

// ============================================================
// Helper: 64-bit magic-div for 10-16 bit integer paths.
// N = 32 + magic.shift (compile-time constant).
// Computes (v * mdiv) >> N using vmull_u32 (u32 x u32 -> u64).
// ============================================================
template<int N>
static AVS_FORCEINLINE uint32x4_t magic_div_wide_neon(uint32x4_t v, uint32_t mdiv) {
  const uint32x2_t mdiv2 = vdup_n_u32(mdiv);
  uint64x2_t lo = vmull_u32(vget_low_u32(v), mdiv2);
  uint64x2_t hi = vmull_u32(vget_high_u32(v), mdiv2);
  lo = vshrq_n_u64(lo, N);
  hi = vshrq_n_u64(hi, N);
  return vcombine_u32(vmovn_u64(lo), vmovn_u64(hi));
}

// ============================================================
// weighted_merge_neon
// No mask.  kernel: (p1*invweight + p2*weight + 16384) >> 15
// weight + invweight == 32768
// ============================================================
void weighted_merge_neon(BYTE* p1, const BYTE* p2,
  int p1_pitch, int p2_pitch,
  int width, int height,
  int weight, int invweight,
  int bits_per_pixel)
{
  const uint16_t w   = (uint16_t)weight;
  const uint16_t inv = (uint16_t)invweight;
  const uint32x4_t round32 = vdupq_n_u32(16384);

  if (bits_per_pixel == 8) {
    // 16 pixels per NEON iteration
    const int vec_width = (width / 16) * 16;
    for (int y = 0; y < height; y++) {
      const uint8_t* srcp2 = reinterpret_cast<const uint8_t*>(p2);
      uint8_t*       dstp  = reinterpret_cast<uint8_t*>(p1);

      for (int x = 0; x < vec_width; x += 16) {
        uint8x16_t v1 = vld1q_u8(dstp  + x);
        uint8x16_t v2 = vld1q_u8(srcp2 + x);

        uint16x8_t p1_lo = vmovl_u8(vget_low_u8(v1));
        uint16x8_t p2_lo = vmovl_u8(vget_low_u8(v2));
        uint16x8_t p1_hi = vmovl_high_u8(v1);
        uint16x8_t p2_hi = vmovl_high_u8(v2);

        // (p1*inv + 16384) + p2*w, then >> 15  — 4 pixels at a time
        uint32x4_t r0 = vshrq_n_u32(vmlal_n_u16(vaddq_u32(vmull_n_u16(vget_low_u16(p1_lo),  inv), round32), vget_low_u16(p2_lo),  w), 15);
        uint32x4_t r1 = vshrq_n_u32(vmlal_n_u16(vaddq_u32(vmull_n_u16(vget_high_u16(p1_lo), inv), round32), vget_high_u16(p2_lo), w), 15);
        uint32x4_t r2 = vshrq_n_u32(vmlal_n_u16(vaddq_u32(vmull_n_u16(vget_low_u16(p1_hi),  inv), round32), vget_low_u16(p2_hi),  w), 15);
        uint32x4_t r3 = vshrq_n_u32(vmlal_n_u16(vaddq_u32(vmull_n_u16(vget_high_u16(p1_hi), inv), round32), vget_high_u16(p2_hi), w), 15);

        uint8x16_t result = vcombine_u8(
          vmovn_u16(vcombine_u16(vmovn_u32(r0), vmovn_u32(r1))),
          vmovn_u16(vcombine_u16(vmovn_u32(r2), vmovn_u32(r3))));
        vst1q_u8(dstp + x, result);
      }
      for (int x = vec_width; x < width; x++)
        dstp[x] = (uint8_t)(((uint32_t)dstp[x] * invweight + (uint32_t)srcp2[x] * weight + 16384) >> 15);

      p1 += p1_pitch;
      p2 += p2_pitch;
    }
  } else {
    // 10-16 bit: 8 pixels per NEON iteration
    const int vec_width = (width / 8) * 8;
    for (int y = 0; y < height; y++) {
      const uint16_t* srcp2 = reinterpret_cast<const uint16_t*>(p2);
      uint16_t*       dstp  = reinterpret_cast<uint16_t*>(p1);

      for (int x = 0; x < vec_width; x += 8) {
        uint16x8_t v1 = vld1q_u16(dstp  + x);
        uint16x8_t v2 = vld1q_u16(srcp2 + x);

        uint32x4_t r0 = vshrq_n_u32(vmlal_n_u16(vaddq_u32(vmull_n_u16(vget_low_u16(v1),  inv), round32), vget_low_u16(v2),  w), 15);
        uint32x4_t r1 = vshrq_n_u32(vmlal_n_u16(vaddq_u32(vmull_n_u16(vget_high_u16(v1), inv), round32), vget_high_u16(v2), w), 15);

        vst1q_u16(dstp + x, vcombine_u16(vmovn_u32(r0), vmovn_u32(r1)));
      }
      for (int x = vec_width; x < width; x++)
        dstp[x] = (uint16_t)(((uint32_t)dstp[x] * invweight + (uint32_t)srcp2[x] * weight + 16384) >> 15);

      p1 += p1_pitch;
      p2 += p2_pitch;
    }
  }
}

// ============================================================
// weighted_merge_float_neon
// No mask.  kernel: p1*(1-weight_f) + p2*weight_f
// ============================================================
void weighted_merge_float_neon(BYTE* p1, const BYTE* p2,
  int p1_pitch, int p2_pitch,
  int width, int height,
  float weight_f)
{
  const float       invweight_f = 1.0f - weight_f;
  const float32x4_t w_v   = vdupq_n_f32(weight_f);
  const float32x4_t inv_v = vdupq_n_f32(invweight_f);

  const int vec_width = (width / 4) * 4;
  for (int y = 0; y < height; y++) {
    const float* srcp2 = reinterpret_cast<const float*>(p2);
    float*       dstp  = reinterpret_cast<float*>(p1);

    for (int x = 0; x < vec_width; x += 4) {
      float32x4_t v1 = vld1q_f32(dstp  + x);
      float32x4_t v2 = vld1q_f32(srcp2 + x);
      // p1 + (p2-p1)*w  ==  p1*inv + p2*w
      vst1q_f32(dstp + x, vmlaq_f32(vmulq_f32(v1, inv_v), v2, w_v));
    }
    for (int x = vec_width; x < width; x++)
      dstp[x] = dstp[x] * invweight_f + srcp2[x] * weight_f;

    p1 += p1_pitch;
    p2 += p2_pitch;
  }
}

// ============================================================
// masked_merge_neon_impl<maskMode, bits_per_pixel>
//
// Mirrors masked_merge_avx2_impl / masked_merge_impl_c in structure.
//
// MaskMode controls mask subsampling (MASK444 = plane-resolution,
// MASK420/MASK422/etc. = luma-resolution; prepare_effective_mask_for_row
// is called from blend_common.h to average the mask per row before blending).
//
// Subtract is handled upstream by pre-inverting the overlay (Layer::Create / Overlay).
// No subtract template dimension here.
//
// 8-bit:   16-bit intermediate magic-div (mulhi-u16 style)
// 10-16bit: 64-bit intermediate magic-div (vmull_u32 -> u64)
// ============================================================

template<MaskMode maskMode, int bits_per_pixel>
static void masked_merge_neon_impl(
  BYTE* p1, const BYTE* p2, const BYTE* mask,
  int p1_pitch, int p2_pitch, int mask_pitch,
  int width, int height, int opacity)
{
  static_assert(bits_per_pixel == 8 || bits_per_pixel == 10 ||
                bits_per_pixel == 12 || bits_per_pixel == 14 ||
                bits_per_pixel == 16, "unsupported bit depth");

  using pixel_t = std::conditional_t<bits_per_pixel == 8, uint8_t, uint16_t>;
  constexpr int      max_val = (1 << bits_per_pixel) - 1;
  constexpr int      half    = max_val / 2;
  constexpr MagicDiv magic   = get_magic_div(bits_per_pixel);

  const pixel_t* maskp         = reinterpret_cast<const pixel_t*>(mask);
  const int      mask_pitch_px = mask_pitch / (int)sizeof(pixel_t);

  // Buffer for non-MASK444 modes: holds one row of averaged mask values.
  std::vector<pixel_t> eff_mask_buf;
  if constexpr (maskMode != MASK444)
    eff_mask_buf.resize(width);

  if constexpr (bits_per_pixel == 8) {
    // --- 8-bit path: 16-bit intermediate magic-div ---
    constexpr uint16_t magic16 = (uint16_t)magic.div;
    constexpr int      mshift  = (int)(16 + magic.shift);  // 23

    const uint16_t   opacity_u16 = (uint16_t)opacity;
    const uint32x4_t half32      = vdupq_n_u32((uint32_t)half);
    const uint32x4_t max32       = vdupq_n_u32((uint32_t)max_val);

    const int vec_width = (width / 8) * 8;
    for (int y = 0; y < height; y++) {
      // Pre-compute effective mask row (no-op for MASK444: returns original pointer)
      const pixel_t* eff_maskp = prepare_effective_mask_for_row<maskMode, pixel_t>(
        maskp, mask_pitch_px, width, eff_mask_buf);

      const uint8_t* eff_mask = reinterpret_cast<const uint8_t*>(eff_maskp);
      const uint8_t* srcp2    = reinterpret_cast<const uint8_t*>(p2);
      uint8_t*       dstp     = reinterpret_cast<uint8_t*>(p1);

      for (int x = 0; x < vec_width; x += 8) {
        uint16x8_t mask16v = vmovl_u8(vld1_u8(eff_mask + x));
        uint16x8_t p1_16   = vmovl_u8(vld1_u8(dstp    + x));
        uint16x8_t p2_raw  = vmovl_u8(vld1_u8(srcp2   + x));

        const uint16x8_t p2_16 = p2_raw;

        // Step 1: m = magic_div(mask * opacity + half)
        // tmp1 = mask*opacity + half  (max 255*255+127 = 65152 fits in u16)
        uint32x4_t tmp1_lo = vaddq_u32(vmull_n_u16(vget_low_u16(mask16v),  opacity_u16), half32);
        uint32x4_t tmp1_hi = vaddq_u32(vmull_n_u16(vget_high_u16(mask16v), opacity_u16), half32);
        uint16x4_t t1_lo   = vmovn_u32(tmp1_lo);
        uint16x4_t t1_hi   = vmovn_u32(tmp1_hi);
        uint32x4_t m_lo    = vshrq_n_u32(vmull_n_u16(t1_lo, magic16), mshift);
        uint32x4_t m_hi    = vshrq_n_u32(vmull_n_u16(t1_hi, magic16), mshift);

        // inv = max - m
        uint32x4_t inv_lo = vsubq_u32(max32, m_lo);
        uint32x4_t inv_hi = vsubq_u32(max32, m_hi);

        // Widen p1, p2 to u32
        uint32x4_t p1_lo32 = vmovl_u16(vget_low_u16(p1_16));
        uint32x4_t p1_hi32 = vmovl_u16(vget_high_u16(p1_16));
        uint32x4_t p2_lo32 = vmovl_u16(vget_low_u16(p2_16));
        uint32x4_t p2_hi32 = vmovl_u16(vget_high_u16(p2_16));

        // Step 2: result = magic_div(p1*inv + p2*m + half)
        // p1*inv + p2*m ≤ max_val^2 (inv+m == max_val) → fits in u32; safe to narrow to u16
        uint32x4_t tmp2_lo = vaddq_u32(vaddq_u32(vmulq_u32(p1_lo32, inv_lo), vmulq_u32(p2_lo32, m_lo)), half32);
        uint32x4_t tmp2_hi = vaddq_u32(vaddq_u32(vmulq_u32(p1_hi32, inv_hi), vmulq_u32(p2_hi32, m_hi)), half32);
        uint16x4_t t2_lo   = vmovn_u32(tmp2_lo);
        uint16x4_t t2_hi   = vmovn_u32(tmp2_hi);
        uint32x4_t res_lo  = vshrq_n_u32(vmull_n_u16(t2_lo, magic16), mshift);
        uint32x4_t res_hi  = vshrq_n_u32(vmull_n_u16(t2_hi, magic16), mshift);

        vst1_u8(dstp + x, vmovn_u16(vcombine_u16(vmovn_u32(res_lo), vmovn_u32(res_hi))));
      }

      // Scalar tail
      for (int x = vec_width; x < width; x++) {
        const uint32_t mval = eff_maskp[x];
        const uint32_t t1   = mval * (uint32_t)opacity + (uint32_t)half;
        const uint32_t m    = (t1 * (uint32_t)magic16) >> mshift;
        const uint32_t inv  = (uint32_t)max_val - m;
        const uint32_t b    = (uint32_t)srcp2[x];
        const uint32_t t2   = (uint32_t)dstp[x] * inv + b * m + (uint32_t)half;
        dstp[x] = (uint8_t)((t2 * (uint32_t)magic16) >> mshift);
      }

      p1   += p1_pitch;
      p2   += p2_pitch;
      // 4:2:0 modes: each chroma output row maps to 2 luma mask rows
      if constexpr (maskMode == MASK420 || maskMode == MASK420_MPEG2 || maskMode == MASK420_TOPLEFT)
        maskp += mask_pitch_px * 2;
      else
        maskp += mask_pitch_px;
    }

  } else {
    // --- 10-16 bit path: 64-bit magic-div via vmull_u32 ---
    constexpr uint32_t magic32     = magic.div;
    constexpr int      total_shift = 32 + (int)magic.shift;

    const uint32_t   opacity_u32 = (uint32_t)opacity;
    const uint32x4_t half32_v    = vdupq_n_u32((uint32_t)half);
    const uint32x4_t max32_v     = vdupq_n_u32((uint32_t)max_val);

    const int vec_width = (width / 4) * 4;
    for (int y = 0; y < height; y++) {
      const pixel_t* eff_maskp = prepare_effective_mask_for_row<maskMode, pixel_t>(
        maskp, mask_pitch_px, width, eff_mask_buf);

      const uint16_t* eff_mask = reinterpret_cast<const uint16_t*>(eff_maskp);
      const uint16_t* srcp2    = reinterpret_cast<const uint16_t*>(p2);
      uint16_t*       dstp     = reinterpret_cast<uint16_t*>(p1);

      for (int x = 0; x < vec_width; x += 4) {
        uint32x4_t mask32v = vmovl_u16(vld1_u16(eff_mask + x));
        uint32x4_t p1_32   = vmovl_u16(vld1_u16(dstp    + x));
        uint32x4_t p2_raw  = vmovl_u16(vld1_u16(srcp2   + x));

        const uint32x4_t p2_32 = p2_raw;

        // tmp1 = mask * opacity + half  (fits in u32 for all depths ≤ 16)
        uint32x4_t tmp1 = vaddq_u32(vmulq_n_u32(mask32v, opacity_u32), half32_v);

        // m = magic_div_wide<total_shift>(tmp1)
        uint32x4_t m   = magic_div_wide_neon<total_shift>(tmp1, magic32);
        uint32x4_t inv = vsubq_u32(max32_v, m);

        // tmp2 = p1*inv + p2*m + half  (bounded by max_val^2 + half < 2^32)
        uint32x4_t tmp2 = vaddq_u32(
          vaddq_u32(vmulq_u32(p1_32, inv), vmulq_u32(p2_32, m)),
          half32_v);

        uint32x4_t result = magic_div_wide_neon<total_shift>(tmp2, magic32);
        vst1_u16(dstp + x, vmovn_u32(result));
      }

      // Scalar tail
      for (int x = vec_width; x < width; x++) {
        const uint32_t mval = eff_maskp[x];
        const uint32_t t1   = mval * opacity_u32 + (uint32_t)half;
        const uint32_t m    = (uint32_t)(((uint64_t)t1 * magic32) >> total_shift);
        const uint32_t inv  = (uint32_t)max_val - m;
        const uint32_t b    = (uint32_t)srcp2[x];
        const uint32_t t2   = (uint32_t)dstp[x] * inv + b * m + (uint32_t)half;
        dstp[x] = (uint16_t)(((uint64_t)t2 * magic32) >> total_shift);
      }

      p1   += p1_pitch;
      p2   += p2_pitch;
      if constexpr (maskMode == MASK420 || maskMode == MASK420_MPEG2 || maskMode == MASK420_TOPLEFT)
        maskp += mask_pitch_px * 2;
      else
        maskp += mask_pitch_px;
    }
  }
}

// ---------------------------------------------------------------------------
// Public dispatch — bits_per_pixel resolved at runtime.
// Mirrors masked_merge_avx2_impl / masked_merge_dispatch_c in structure.
// Call this from Layer with the appropriate MaskMode.
// ---------------------------------------------------------------------------

template<MaskMode maskMode>
void masked_merge_neon_dispatch(BYTE* p1, const BYTE* p2, const BYTE* mask,
  int p1_pitch, int p2_pitch, int mask_pitch,
  int width, int height, int opacity, int bits_per_pixel)
{
  switch (bits_per_pixel) {
  case  8: masked_merge_neon_impl<maskMode,  8>(p1, p2, mask, p1_pitch, p2_pitch, mask_pitch, width, height, opacity); break;
  case 10: masked_merge_neon_impl<maskMode, 10>(p1, p2, mask, p1_pitch, p2_pitch, mask_pitch, width, height, opacity); break;
  case 12: masked_merge_neon_impl<maskMode, 12>(p1, p2, mask, p1_pitch, p2_pitch, mask_pitch, width, height, opacity); break;
  case 14: masked_merge_neon_impl<maskMode, 14>(p1, p2, mask, p1_pitch, p2_pitch, mask_pitch, width, height, opacity); break;
  case 16: masked_merge_neon_impl<maskMode, 16>(p1, p2, mask, p1_pitch, p2_pitch, mask_pitch, width, height, opacity); break;
  }
}

// ---------------------------------------------------------------------------
// Public Overlay API — fixed MASK444 (mask already at plane resolution).
// Mirrors masked_merge_avx2 / masked_merge_c calling their MASK444 impl.
// ---------------------------------------------------------------------------

void masked_merge_neon(BYTE* p1, const BYTE* p2, const BYTE* mask,
  int p1_pitch, int p2_pitch, int mask_pitch,
  int width, int height, int opacity, int bits_per_pixel)
{
  masked_merge_neon_dispatch<MASK444>(
    p1, p2, mask, p1_pitch, p2_pitch, mask_pitch, width, height, opacity, bits_per_pixel);
}

// ---------------------------------------------------------------------------
// Explicit instantiations of masked_merge_neon_dispatch.
// Overlay path needs MASK444.  Layer support (future) needs the other modes.
// ---------------------------------------------------------------------------

// One instantiation per MaskMode — no is_chroma dimension (subtract removed, formula identical for luma and chroma).
template void masked_merge_neon_dispatch<MASK444>       (BYTE*, const BYTE*, const BYTE*, int, int, int, int, int, int, int);
template void masked_merge_neon_dispatch<MASK411>       (BYTE*, const BYTE*, const BYTE*, int, int, int, int, int, int, int);
template void masked_merge_neon_dispatch<MASK420>       (BYTE*, const BYTE*, const BYTE*, int, int, int, int, int, int, int);
template void masked_merge_neon_dispatch<MASK420_MPEG2> (BYTE*, const BYTE*, const BYTE*, int, int, int, int, int, int, int);
template void masked_merge_neon_dispatch<MASK420_TOPLEFT>(BYTE*, const BYTE*, const BYTE*, int, int, int, int, int, int, int);
template void masked_merge_neon_dispatch<MASK422>       (BYTE*, const BYTE*, const BYTE*, int, int, int, int, int, int, int);
template void masked_merge_neon_dispatch<MASK422_MPEG2> (BYTE*, const BYTE*, const BYTE*, int, int, int, int, int, int, int);
template void masked_merge_neon_dispatch<MASK422_TOPLEFT>(BYTE*, const BYTE*, const BYTE*, int, int, int, int, int, int, int);

// ============================================================
// masked_merge_float_neon
// Float masked blend.  kernel: p1 + (p2-p1)*(mask*opacity)
// ============================================================
void masked_merge_float_neon(BYTE* p1, const BYTE* p2, const BYTE* mask,
  int p1_pitch, int p2_pitch, int mask_pitch,
  int width, int height, float opacity_f)
{
  const float32x4_t opacity_v = vdupq_n_f32(opacity_f);
  const int vec_width = (width / 4) * 4;

  for (int y = 0; y < height; y++) {
    const float* maskp = reinterpret_cast<const float*>(mask);
    const float* srcp2 = reinterpret_cast<const float*>(p2);
    float*       dstp  = reinterpret_cast<float*>(p1);

    for (int x = 0; x < vec_width; x += 4) {
      float32x4_t v_mask = vmulq_f32(vld1q_f32(maskp + x), opacity_v);
      float32x4_t v_p1   = vld1q_f32(dstp  + x);
      float32x4_t v_p2   = vld1q_f32(srcp2 + x);
      // p1 + (p2 - p1) * (mask * opacity)
      vst1q_f32(dstp + x, vmlaq_f32(v_p1, vsubq_f32(v_p2, v_p1), v_mask));
    }
    for (int x = vec_width; x < width; x++) {
      const float m = maskp[x] * opacity_f;
      dstp[x] = dstp[x] + (srcp2[x] - dstp[x]) * m;
    }

    p1   += p1_pitch;
    p2   += p2_pitch;
    mask += mask_pitch;
  }
}

/********************************
 ********* Blend Opaque *********
 ** Use for Lighten and Darken **
 ********************************/

// NEON equivalent of overlay_blend_opaque_sse2_core
AVS_FORCEINLINE uint8x16_t overlay_blend_opaque_neon_core(const uint8x16_t& p1, const uint8x16_t& p2, const uint8x16_t& mask) {
  return vbslq_u8(mask, p2, p1);
}

// For 16-bit pixels
AVS_FORCEINLINE uint16x8_t overlay_blend_opaque_neon_core_u16(const uint16x8_t& p1, const uint16x8_t& p2, const uint16x8_t& mask) {
  return vbslq_u16(mask, p2, p1);
}

// Compare functions for lighten and darken mode (8-bit)
AVS_FORCEINLINE static uint8x16_t overlay_darken_neon_cmp(const uint8x16_t& p1, const uint8x16_t& p2) {
  return vorrq_u8(vcltq_u8(p2, p1), vceqq_u8(p2, p1));
}

AVS_FORCEINLINE static uint8x16_t overlay_lighten_neon_cmp(const uint8x16_t& p1, const uint8x16_t& p2) {
  return vorrq_u8(vcgtq_u8(p2, p1), vceqq_u8(p2, p1));
}

AVS_FORCEINLINE static int overlay_darken_c_cmp(BYTE p1, BYTE p2) { return p2 <= p1; }
AVS_FORCEINLINE static int overlay_lighten_c_cmp(BYTE p1, BYTE p2) { return p2 >= p1; }

/***************************************
 ********* Mode: Lighten/Darken ********
 ***************************************/

using OverlayNeonCompare = uint8x16_t(const uint8x16_t& p1, const uint8x16_t& p2);
using OverlayCCompare = int(BYTE, BYTE);

template <OverlayNeonCompare compare, OverlayCCompare compare_c>
void overlay_darklighten_neon(BYTE *p1Y, BYTE *p1U, BYTE *p1V, const BYTE *p2Y, const BYTE *p2U, const BYTE *p2V,
                              int p1_pitch, int p2_pitch, int width, int height) {
  int wMod16 = (width / 16) * 16;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < wMod16; x += 16) {
      uint8x16_t p1_y = vld1q_u8(p1Y + x);
      uint8x16_t p2_y = vld1q_u8(p2Y + x);
      uint8x16_t mask = compare(p1_y, p2_y);
      vst1q_u8(p1Y + x, overlay_blend_opaque_neon_core(p1_y, p2_y, mask));

      uint8x16_t p1_u = vld1q_u8(p1U + x);
      uint8x16_t p2_u = vld1q_u8(p2U + x);
      vst1q_u8(p1U + x, overlay_blend_opaque_neon_core(p1_u, p2_u, mask));

      uint8x16_t p1_v = vld1q_u8(p1V + x);
      uint8x16_t p2_v = vld1q_u8(p2V + x);
      vst1q_u8(p1V + x, overlay_blend_opaque_neon_core(p1_v, p2_v, mask));
    }
    for (int x = wMod16; x < width; x++) {
      int mask = compare_c(p1Y[x], p2Y[x]) ? 0xFF : 0x00;
      p1Y[x] = overlay_blend_opaque_c_core<uint8_t>(p1Y[x], p2Y[x], mask);
      p1U[x] = overlay_blend_opaque_c_core<uint8_t>(p1U[x], p2U[x], mask);
      p1V[x] = overlay_blend_opaque_c_core<uint8_t>(p1V[x], p2V[x], mask);
    }
    p1Y += p1_pitch; p1U += p1_pitch; p1V += p1_pitch;
    p2Y += p2_pitch; p2U += p2_pitch; p2V += p2_pitch;
  }
}

void overlay_darken_neon(BYTE* p1Y, BYTE* p1U, BYTE* p1V, const BYTE* p2Y, const BYTE* p2U, const BYTE* p2V, int p1_pitch, int p2_pitch, int width, int height) {
  overlay_darklighten_neon<overlay_darken_neon_cmp, overlay_darken_c_cmp>(p1Y, p1U, p1V, p2Y, p2U, p2V, p1_pitch, p2_pitch, width, height);
}
void overlay_lighten_neon(BYTE* p1Y, BYTE* p1U, BYTE* p1V, const BYTE* p2Y, const BYTE* p2U, const BYTE* p2V, int p1_pitch, int p2_pitch, int width, int height) {
  overlay_darklighten_neon<overlay_lighten_neon_cmp, overlay_lighten_c_cmp>(p1Y, p1U, p1V, p2Y, p2U, p2V, p1_pitch, p2_pitch, width, height);
}

// ---------------------------------------------------------------------------
// Overlay blend masked getter — returns masked_merge_neon_dispatch instantiation.
// is_chroma=false → always MASK444 (luma).
// is_chroma=true  → placement-aware maskMode (chroma).
// ---------------------------------------------------------------------------
masked_merge_fn_t* get_overlay_blend_masked_fn_neon(bool is_chroma, MaskMode maskMode)
{
#define DISPATCH_OVERLAY_BLEND_NEON(MaskType) \
  return is_chroma ? masked_merge_neon_dispatch<MaskType> \
                   : masked_merge_neon_dispatch<MASK444>;

  switch (maskMode) {
  case MASK444:          DISPATCH_OVERLAY_BLEND_NEON(MASK444)
  case MASK420:          DISPATCH_OVERLAY_BLEND_NEON(MASK420)
  case MASK420_MPEG2:    DISPATCH_OVERLAY_BLEND_NEON(MASK420_MPEG2)
  case MASK420_TOPLEFT:  DISPATCH_OVERLAY_BLEND_NEON(MASK420_TOPLEFT)
  case MASK422:          DISPATCH_OVERLAY_BLEND_NEON(MASK422)
  case MASK422_MPEG2:    DISPATCH_OVERLAY_BLEND_NEON(MASK422_MPEG2)
  case MASK422_TOPLEFT:  DISPATCH_OVERLAY_BLEND_NEON(MASK422_TOPLEFT)
  case MASK411:          DISPATCH_OVERLAY_BLEND_NEON(MASK411)
  }
#undef DISPATCH_OVERLAY_BLEND_NEON
  return masked_merge_neon_dispatch<MASK444>; // unreachable
}
