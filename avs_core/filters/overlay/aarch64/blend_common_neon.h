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

#ifndef __blend_common_neon_h
#define __blend_common_neon_h

#include <avs/types.h>
#include "../blend_common.h"  // MaskMode, masked_merge_fn_t, weighted_merge_fn_t, ...

// ============================================================
// Family 1: weighted merge — no mask, weight+invweight == 32768
// kernel: (p1*invweight + p2*weight + 16384) >> 15
// ============================================================
void weighted_merge_neon(BYTE* p1, const BYTE* p2,
  int p1_pitch, int p2_pitch,
  int width, int height,
  int weight, int invweight,
  int bits_per_pixel);

void weighted_merge_float_neon(BYTE* p1, const BYTE* p2,
  int p1_pitch, int p2_pitch,
  int width, int height,
  float weight_f);

// ============================================================
// Family 2: masked merge — integer, magic-div normalization.
//
// masked_merge_neon_dispatch<maskMode, is_chroma>
//   Full dispatch used by Overlay (MASK444) and Layer (MASK420/MASK422/…).
//   opacity pre-scaled: round(opacity_f * max_pixel_value).
//   Subtract is handled by pre-inverting the overlay before calling (Layer::Create /
//   Overlay).  No subtract template dimension needed here.
//   Explicit instantiations for all MaskMode × is_chroma
//   combinations are provided in blend_common_neon.cpp.
//
// masked_merge_neon — convenience entry point fixing MASK444, is_chroma=false.
//   Mirrors masked_merge_avx2 / masked_merge_c calling their MASK444 impl.
// ============================================================

// Template declaration — definition + explicit instantiations in .cpp.
template<MaskMode maskMode>
void masked_merge_neon_dispatch(BYTE* p1, const BYTE* p2, const BYTE* mask,
  int p1_pitch, int p2_pitch, int mask_pitch,
  int width, int height,
  int opacity, int bits_per_pixel);

// MASK444 convenience (used by OF_blend.cpp Overlay path)
void masked_merge_neon(BYTE* p1, const BYTE* p2, const BYTE* mask,
  int p1_pitch, int p2_pitch, int mask_pitch,
  int width, int height,
  int opacity, int bits_per_pixel);

// Float masked blend
void masked_merge_float_neon(BYTE* p1, const BYTE* p2, const BYTE* mask,
  int p1_pitch, int p2_pitch, int mask_pitch,
  int width, int height,
  float opacity_f);

// ============================================================
// Mode: Darken/Lighten (8-bit only)
// ============================================================
void overlay_darken_neon(BYTE* p1Y, BYTE* p1U, BYTE* p1V, const BYTE* p2Y, const BYTE* p2U, const BYTE* p2V, int p1_pitch, int p2_pitch, int width, int height);
void overlay_lighten_neon(BYTE* p1Y, BYTE* p1U, BYTE* p1V, const BYTE* p2Y, const BYTE* p2U, const BYTE* p2V, int p1_pitch, int p2_pitch, int width, int height);

// ============================================================
// Overlay blend masked getter.
// Returns masked_merge_neon_dispatch instantiation for the given is_chroma / maskMode.
// Overlay always passes MASK444 (internal format is YUV444).
// ============================================================
masked_merge_fn_t* get_overlay_blend_masked_fn_neon(bool is_chroma, MaskMode maskMode);

#endif // __blend_common_neon_h
