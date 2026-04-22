// AviSynth+  Copyright 2026- AviSynth+ Project
// SPDX-License-Identifier: GPL-2.0-or-later

// NEON Layer add dispatcher — mirrors layer_sse41.h / layer_avx2.h.
// Only integer hasAlpha=true use_chroma=true paths are accelerated;
// float/hasAlpha=false/use_chroma=false fall through to C templates.
// subtract is handled by pre-inverting the overlay in Layer::Create.

#ifndef __Layer_NEON_H__
#define __Layer_NEON_H__

#include <avisynth.h>
#include "../../layer.h"   // layer_yuv_add_c_t, layer_yuv_add_f_c_t,
                           // layer_planarrgb_add_c_t, layer_planarrgb_add_f_c_t

void get_layer_yuv_add_functions_neon(
  bool is_chroma, bool use_chroma, bool hasAlpha,
  int placement, VideoInfo& vi, int bits_per_pixel,
  layer_yuv_add_c_t** layer_fn,
  layer_yuv_add_f_c_t** layer_f_fn);

void get_layer_planarrgb_add_functions_neon(
  bool chroma, bool hasAlpha, bool blendAlpha, int bits_per_pixel,
  layer_planarrgb_add_c_t** layer_fn,
  layer_planarrgb_add_f_c_t** layer_f_fn);

#endif  // __Layer_NEON_H__
