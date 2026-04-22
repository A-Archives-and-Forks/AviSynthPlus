// AviSynth+  Copyright 2026- AviSynth+ Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// NEON Layer add/subtract dispatcher.
// Named *_neon.cpp so the CMake handle_arch_flags(NEON) glob assigns
// per-file -march=armv8-a (GCC/Clang) flags to this translation unit.
// Subtract is handled upstream by pre-inverting the overlay in Layer::Create;
// GetFrame never sees Op="Subtract" for paths dispatched here.

#include "../../layer.h"       // layer_yuv_add_c_t/f, PLACEMENT_*, pulls in blend_common.h
#include "layer_neon.h"        // declarations

#include <arm_neon.h>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "blend_common_neon.h"  // masked_merge_neon_dispatch

#include "../../../core/internal.h"  // DISABLE_WARNING_PUSH/POP — needed by layer.hpp

// C template dispatcher: get_layer_yuv_add_functions / get_layer_planarrgb_add_functions.
// static functions — this TU's copy gets -march=armv8-a auto-vectorization (GCC/Clang).
#include "../../layer.hpp"

// ---------------------------------------------------------------------------
// Planar RGB add — NEON per-plane wrappers (mirrors SSE4.1/AVX2 counterparts).
// All planar RGB planes use MASK444. maskp8 is the per-pixel weight.
// chroma=false and float fall back to C templates.

static void layer_planarrgb_add_neon_3plane(
  BYTE** dstp8, const BYTE** ovrp8, const BYTE* maskp8,
  int dst_pitch, int overlay_pitch, int mask_pitch,
  int width, int height, int level, int bits_per_pixel)
{
  for (int i = 0; i < 3; i++)
    masked_merge_neon_dispatch<MASK444>(
      dstp8[i], ovrp8[i], maskp8,
      dst_pitch, overlay_pitch, mask_pitch,
      width, height, level, bits_per_pixel);
}

static void layer_planarrgb_add_neon_4plane(
  BYTE** dstp8, const BYTE** ovrp8, const BYTE* maskp8,
  int dst_pitch, int overlay_pitch, int mask_pitch,
  int width, int height, int level, int bits_per_pixel)
{
  for (int i = 0; i < 4; i++)
    masked_merge_neon_dispatch<MASK444>(
      dstp8[i], ovrp8[i], maskp8,
      dst_pitch, overlay_pitch, mask_pitch,
      width, height, level, bits_per_pixel);
}

void get_layer_planarrgb_add_functions_neon(
  bool chroma, bool hasAlpha, bool blendAlpha, int bits_per_pixel,
  layer_planarrgb_add_c_t** layer_fn,
  layer_planarrgb_add_f_c_t** layer_f_fn)
{
  if (hasAlpha && chroma && bits_per_pixel != 32) {
    *layer_fn = blendAlpha ? layer_planarrgb_add_neon_4plane : layer_planarrgb_add_neon_3plane;
    return;
  }
  get_layer_planarrgb_add_functions(chroma, hasAlpha, blendAlpha, bits_per_pixel, layer_fn, layer_f_fn);
}

// ---------------------------------------------------------------------------
// NEON Layer YUV add dispatcher.
// Selects masked_merge_neon_dispatch<maskMode> for integer hasAlpha=true,
// use_chroma=true cases; falls through to C templates for float,
// hasAlpha=false, or use_chroma=false (blend-toward-neutral).
// subtract is handled by pre-inverting the overlay in Layer::Create.
// ---------------------------------------------------------------------------
void get_layer_yuv_add_functions_neon(
  bool is_chroma, bool use_chroma, bool hasAlpha,
  int placement, VideoInfo& vi, int bits_per_pixel,
  layer_yuv_add_c_t** layer_fn,
  layer_yuv_add_f_c_t** layer_f_fn)
{
  // Conditions where NEON masked_merge doesn't apply:
  //  - float (no integer NEON path for float)
  //  - no alpha mask (blend uses flat weight, not mask; different arithmetic)
  //  - use_chroma=false with is_chroma=true (blend-toward-neutral, different formula)
  if (bits_per_pixel == 32 || !hasAlpha || (is_chroma && !use_chroma)) {
    get_layer_yuv_add_functions(
      is_chroma, use_chroma, hasAlpha, placement, vi, bits_per_pixel, layer_fn, layer_f_fn);
    return;
  }

  // Determine MaskMode from format and placement
  MaskMode maskMode = MASK444;
  if (is_chroma) {
    if (vi.IsYV411())
      maskMode = MASK411;
    else if (vi.Is420())
      maskMode = (placement == PLACEMENT_MPEG1) ? MASK420 : (placement == PLACEMENT_TOPLEFT) ? MASK420_TOPLEFT : MASK420_MPEG2;
    else if (vi.Is422())
      maskMode = (placement == PLACEMENT_MPEG1) ? MASK422 : (placement == PLACEMENT_TOPLEFT) ? MASK422_TOPLEFT : MASK422_MPEG2;
    // Is444() / IsY(): stay MASK444
  }
  // is_chroma=false (luma): always MASK444

  // Dispatch to the appropriate NEON instantiation.
  // For luma (is_chroma=false): always MASK444, no placement.
  // For chroma (is_chroma=true, use_chroma=true): placement-aware maskMode.
#define DISPATCH_LUMA_CHROMA_NEON(MaskType) \
  if (is_chroma) *layer_fn = masked_merge_neon_dispatch<MaskType>; \
  else           *layer_fn = masked_merge_neon_dispatch<MASK444>;

  switch (maskMode) {
  case MASK444:          DISPATCH_LUMA_CHROMA_NEON(MASK444)          break;
  case MASK420:          DISPATCH_LUMA_CHROMA_NEON(MASK420)          break;
  case MASK420_MPEG2:    DISPATCH_LUMA_CHROMA_NEON(MASK420_MPEG2)    break;
  case MASK420_TOPLEFT:  DISPATCH_LUMA_CHROMA_NEON(MASK420_TOPLEFT)  break;
  case MASK422:          DISPATCH_LUMA_CHROMA_NEON(MASK422)          break;
  case MASK422_MPEG2:    DISPATCH_LUMA_CHROMA_NEON(MASK422_MPEG2)    break;
  case MASK422_TOPLEFT:  DISPATCH_LUMA_CHROMA_NEON(MASK422_TOPLEFT)  break;
  case MASK411:          DISPATCH_LUMA_CHROMA_NEON(MASK411)          break;
  }
#undef DISPATCH_LUMA_CHROMA_NEON
}
