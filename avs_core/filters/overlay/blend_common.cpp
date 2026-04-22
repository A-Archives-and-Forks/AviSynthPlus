// Avisynth v2.5.  Copyright 2002 Ben Rudiak-Gould et al.
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

// Overlay (c) 2003, 2004 by Klaus Post

#include <avs/config.h>

#include "blend_common.h"
#include "overlayfunctions.h"

#include <stdint.h>
#include <type_traits>


/******************************
 ********* Mode: Blend ********
 ******************************/

// ---------------------------------------------------------------------------
// Static C wrappers for Overlay blend chroma with non-MASK444 modes.
// masked_merge_dispatch_c is AVS_FORCEINLINE static — no external address.
// These thin wrappers give addressable entry points for the getter below.
// MASK444 chroma reuses masked_merge_c.
// ---------------------------------------------------------------------------
#define BLEND_C_CHROMA_WRAP(suffix, MaskType) \
  static void masked_merge_c_chroma_##suffix( \
    BYTE* p1, const BYTE* p2, const BYTE* mask, \
    int p1_pitch, int p2_pitch, int mask_pitch, \
    int width, int height, int opacity, int bits_per_pixel) { \
    masked_merge_dispatch_c<MaskType>( \
      p1, p2, mask, p1_pitch, p2_pitch, mask_pitch, width, height, opacity, bits_per_pixel); \
  }
BLEND_C_CHROMA_WRAP(mask411,           MASK411)
BLEND_C_CHROMA_WRAP(mask420,           MASK420)
BLEND_C_CHROMA_WRAP(mask420_mpeg2,     MASK420_MPEG2)
BLEND_C_CHROMA_WRAP(mask420_topleft,   MASK420_TOPLEFT)
BLEND_C_CHROMA_WRAP(mask422,           MASK422)
BLEND_C_CHROMA_WRAP(mask422_mpeg2,     MASK422_MPEG2)
BLEND_C_CHROMA_WRAP(mask422_topleft,   MASK422_TOPLEFT)
#undef BLEND_C_CHROMA_WRAP

masked_merge_fn_t* get_overlay_blend_masked_fn_c(bool is_chroma, MaskMode maskMode) {
  // Luma (is_chroma=false) is always MASK444; reuses masked_merge_c.
  // Chroma MASK444 also reuses masked_merge_c (subtract=false → is_chroma irrelevant).
  if (!is_chroma || maskMode == MASK444)
    return &masked_merge_c;
  switch (maskMode) {
  case MASK411:          return &masked_merge_c_chroma_mask411;
  case MASK420:          return &masked_merge_c_chroma_mask420;
  case MASK420_MPEG2:    return &masked_merge_c_chroma_mask420_mpeg2;
  case MASK420_TOPLEFT:  return &masked_merge_c_chroma_mask420_topleft;
  case MASK422:          return &masked_merge_c_chroma_mask422;
  case MASK422_MPEG2:    return &masked_merge_c_chroma_mask422_mpeg2;
  case MASK422_TOPLEFT:  return &masked_merge_c_chroma_mask422_topleft;
  default:               return &masked_merge_c; // unreachable
  }
}

// ---------------------------------------------------------------------------
// Float C wrappers — same structure as integer wrappers above.
// MASK444 (any is_chroma) reuses masked_merge_float_c.
// ---------------------------------------------------------------------------
#define BLEND_FLOAT_C_CHROMA_WRAP(suffix, MaskType) \
  static void masked_merge_float_c_chroma_##suffix( \
    BYTE* p1, const BYTE* p2, const BYTE* mask, \
    int p1_pitch, int p2_pitch, int mask_pitch, \
    int width, int height, float opacity_f) { \
    masked_merge_impl_float_c<MaskType>( \
      p1, p2, mask, p1_pitch, p2_pitch, mask_pitch, width, height, opacity_f); \
  }
BLEND_FLOAT_C_CHROMA_WRAP(mask411,           MASK411)
BLEND_FLOAT_C_CHROMA_WRAP(mask420,           MASK420)
BLEND_FLOAT_C_CHROMA_WRAP(mask420_mpeg2,     MASK420_MPEG2)
BLEND_FLOAT_C_CHROMA_WRAP(mask420_topleft,   MASK420_TOPLEFT)
BLEND_FLOAT_C_CHROMA_WRAP(mask422,           MASK422)
BLEND_FLOAT_C_CHROMA_WRAP(mask422_mpeg2,     MASK422_MPEG2)
BLEND_FLOAT_C_CHROMA_WRAP(mask422_topleft,   MASK422_TOPLEFT)
#undef BLEND_FLOAT_C_CHROMA_WRAP

masked_merge_float_fn_t* get_overlay_blend_masked_float_fn_c(bool is_lumamask_based_chroma, MaskMode maskMode) {
  if (!is_lumamask_based_chroma || maskMode == MASK444)
    return &masked_merge_float_c;
  switch (maskMode) {
  case MASK411:          return &masked_merge_float_c_chroma_mask411;
  case MASK420:          return &masked_merge_float_c_chroma_mask420;
  case MASK420_MPEG2:    return &masked_merge_float_c_chroma_mask420_mpeg2;
  case MASK420_TOPLEFT:  return &masked_merge_float_c_chroma_mask420_topleft;
  case MASK422:          return &masked_merge_float_c_chroma_mask422;
  case MASK422_MPEG2:    return &masked_merge_float_c_chroma_mask422_mpeg2;
  case MASK422_TOPLEFT:  return &masked_merge_float_c_chroma_mask422_topleft;
  default:               return &masked_merge_float_c; // unreachable
  }
}

/*
// Scalar division — used in reference C, constexpr-friendly
template<int bits_per_pixel>
inline int magic_div(uint32_t tmp) {
  constexpr MagicDiv magic = get_magic_div(bits_per_pixel);
  if constexpr (bits_per_pixel == 8)
    // mimics: mulhi_epu16(x, 0x8081) >> 7
    return (int)(((uint32_t)tmp * magic.div) >> (16 + magic.shift));
  else
    // mimics: mul_epu32(x, div) >> (32 + shift)
    return (int)(((uint64_t)tmp * magic.div) >> (32 + magic.shift));
}
*/

/*****************************************************
 ********* Family 1: weighted_merge (no mask) ********
 *****************************************************/

// weight + invweight == 32768; kernel: (p1*inv + p2*w + 16384) >> 15
// Intentionally matches the SIMD >> 15 path (old C used >> 16 with weights summing to 32767).
template<typename pixel_t>
static void weighted_merge_impl_c(BYTE* p1, const BYTE* p2,
  int p1_pitch, int p2_pitch,
  int width, int height,
  int weight, int invweight)
{
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      reinterpret_cast<pixel_t*>(p1)[x] = (pixel_t)(
        (reinterpret_cast<const pixel_t*>(p1)[x] * invweight +
         reinterpret_cast<const pixel_t*>(p2)[x] * weight + 16384) >> 15);
    }
    p1 += p1_pitch;
    p2 += p2_pitch;
  }
}

void weighted_merge_c(BYTE* p1, const BYTE* p2,
  int p1_pitch, int p2_pitch,
  int width, int height,
  int weight, int invweight,
  int bits_per_pixel)
{
  if (bits_per_pixel == 8)
    weighted_merge_impl_c<uint8_t>(p1, p2, p1_pitch, p2_pitch, width, height, weight, invweight);
  else
    weighted_merge_impl_c<uint16_t>(p1, p2, p1_pitch, p2_pitch, width, height, weight, invweight);
}

void weighted_merge_float_c(BYTE* p1, const BYTE* p2,
  int p1_pitch, int p2_pitch,
  int width, int height,
  float weight_f)
{
  const float invweight_f = 1.0f - weight_f;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      reinterpret_cast<float*>(p1)[x] =
        reinterpret_cast<const float*>(p1)[x] * invweight_f +
        reinterpret_cast<const float*>(p2)[x] * weight_f;
    }
    p1 += p1_pitch;
    p2 += p2_pitch;
  }
}

/*************************************************************
 *************** Family 2: masked_merge **********************
 * Implementations are inline templates in blend_common.h.   *
 * Public functions below fix maskMode=MASK444 (mask already *
 * at plane resolution) — the common Overlay path.           *
 * Layer passes other MaskModes via masked_merge_dispatch_c  *
 * or masked_merge_impl_float_c directly.                    *
 ************************************************************/

// Template implementations live in blend_common.h (AVS_FORCEINLINE static).
// These thin wrappers fix maskMode=MASK444 for the Overlay path where the
// mask is already at the processed plane's resolution.
// Layer uses masked_merge_dispatch_c / masked_merge_impl_float_c directly
// with the appropriate MaskMode.

void masked_merge_c(BYTE* p1, const BYTE* p2, const BYTE* mask,
  int p1_pitch, int p2_pitch, int mask_pitch,
  int width, int height, int opacity, int bits_per_pixel)
{
  masked_merge_dispatch_c<MASK444>(
    p1, p2, mask, p1_pitch, p2_pitch, mask_pitch, width, height, opacity, bits_per_pixel);
}

void masked_merge_float_c(BYTE* p1, const BYTE* p2, const BYTE* mask,
  int p1_pitch, int p2_pitch, int mask_pitch,
  int width, int height, float opacity_f)
{
  masked_merge_impl_float_c<MASK444>(
    p1, p2, mask, p1_pitch, p2_pitch, mask_pitch, width, height, opacity_f);
}

/***************************************
 ********* Mode: Lighten/Darken ********
 ***************************************/

typedef int (OverlayCCompare)(BYTE, BYTE);

template<typename pixel_t, bool darken /* OverlayCCompare<pixel_t> compare*/>
static void overlay_darklighten_c(BYTE *p1Y_8, BYTE *p1U_8, BYTE *p1V_8, const BYTE *p2Y_8, const BYTE *p2U_8, const BYTE *p2V_8, int p1_pitch, int p2_pitch, int width, int height) {
  pixel_t* p1Y = reinterpret_cast<pixel_t *>(p1Y_8);
  pixel_t* p1U = reinterpret_cast<pixel_t *>(p1U_8);
  pixel_t* p1V = reinterpret_cast<pixel_t *>(p1V_8);

  const pixel_t* p2Y = reinterpret_cast<const pixel_t *>(p2Y_8);
  const pixel_t* p2U = reinterpret_cast<const pixel_t *>(p2U_8);
  const pixel_t* p2V = reinterpret_cast<const pixel_t *>(p2V_8);

  // pitches are already scaled
  //p1_pitch /= sizeof(pixel_t);
  //p2_pitch /= sizeof(pixel_t);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int mask = darken ? (p2Y[x] <= p1Y[x]) : (p2Y[x] >= p1Y[x]); // compare(p1Y[x], p2Y[x]);
      p1Y[x] = overlay_blend_opaque_c_core<pixel_t>(p1Y[x], p2Y[x], mask);
      p1U[x] = overlay_blend_opaque_c_core<pixel_t>(p1U[x], p2U[x], mask);
      p1V[x] = overlay_blend_opaque_c_core<pixel_t>(p1V[x], p2V[x], mask);
    }

    p1Y += p1_pitch;
    p1U += p1_pitch;
    p1V += p1_pitch;

    p2Y += p2_pitch;
    p2U += p2_pitch;
    p2V += p2_pitch;
  }
}

// Exported function
template<typename pixel_t>
void overlay_darken_c(BYTE *p1Y_8, BYTE *p1U_8, BYTE *p1V_8, const BYTE *p2Y_8, const BYTE *p2U_8, const BYTE *p2V_8, int p1_pitch, int p2_pitch, int width, int height) {
  overlay_darklighten_c<pixel_t, true /*overlay_darken_c_cmp */>(p1Y_8, p1U_8, p1V_8, p2Y_8, p2U_8, p2V_8, p1_pitch, p2_pitch, width, height);
}
// instantiate
template void overlay_darken_c<uint8_t>(BYTE *p1Y_8, BYTE *p1U_8, BYTE *p1V_8, const BYTE *p2Y_8, const BYTE *p2U_8, const BYTE *p2V_8, int p1_pitch, int p2_pitch, int width, int height);
template void overlay_darken_c<uint16_t>(BYTE *p1Y_8, BYTE *p1U_8, BYTE *p1V_8, const BYTE *p2Y_8, const BYTE *p2U_8, const BYTE *p2V_8, int p1_pitch, int p2_pitch, int width, int height);

template<typename pixel_t>
void overlay_lighten_c(BYTE *p1Y_8, BYTE *p1U_8, BYTE *p1V_8, const BYTE *p2Y_8, const BYTE *p2U_8, const BYTE *p2V_8, int p1_pitch, int p2_pitch, int width, int height) {
  overlay_darklighten_c<pixel_t, false /*overlay_lighten_c_cmp*/>(p1Y_8, p1U_8, p1V_8, p2Y_8, p2U_8, p2V_8, p1_pitch, p2_pitch, width, height);
}

// instantiate
template void overlay_lighten_c<uint8_t>(BYTE *p1Y_8, BYTE *p1U_8, BYTE *p1V_8, const BYTE *p2Y_8, const BYTE *p2U_8, const BYTE *p2V_8, int p1_pitch, int p2_pitch, int width, int height);
template void overlay_lighten_c<uint16_t>(BYTE *p1Y_8, BYTE *p1U_8, BYTE *p1V_8, const BYTE *p2Y_8, const BYTE *p2U_8, const BYTE *p2V_8, int p1_pitch, int p2_pitch, int width, int height);
