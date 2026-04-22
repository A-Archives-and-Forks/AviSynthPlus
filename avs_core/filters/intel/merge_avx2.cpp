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


// Avisynth filter: YUV merge / Swap planes
// by Klaus Post (kp@interact.dk)
// adapted by Richard Berg (avisynth-dev@richardberg.net)
// iSSE code by Ian Brabham


// Intrinsics base header + really required extension headers
#if defined(_MSC_VER)
#include <intrin.h> // MSVC
#else 
#include <x86intrin.h> // GCC/MinGW/Clang/LLVM
#endif
#include <immintrin.h>


#if !defined(__FMA__)
// Assume that all processors that have AVX2 also have FMA3
#if defined(__GNUC__) && !defined(__INTEL_COMPILER) && !defined(__clang__)
// Prevent error message in g++ when using FMA intrinsics with avx2:
#pragma message "It is recommended to specify also option -mfma when using -mavx2 or higher"
#else
#define __FMA__  1
#endif
#endif
// FMA3 instruction set
#if defined(__FMA__) && (defined(__GNUC__) || defined(__clang__))  && !defined(__INTEL_COMPILER)
#include <fmaintrin.h>
#endif // __FMA__

#include "merge_avx2.h"
#include <avisynth.h>
#include <avs/types.h>
#include <cstdint>

#ifndef _mm256_set_m128i
#define _mm256_set_m128i(v0, v1) _mm256_insertf128_si256(_mm256_castsi128_si256(v1), (v0), 1)
#endif

#ifndef _mm256_set_m128
#define _mm256_set_m128(v0, v1) _mm256_insertf128_ps(_mm256_castps128_ps256(v1), (v0), 1)
#endif

/* -----------------------------------
 *            average_plane
 * -----------------------------------
 */
#define AVX2BUG_WORKAROUND
// VS2017 15.5.1..2 optimizer generates illegal instructions for vmovntdqa
// just a note, to be removed
template<typename pixel_t>
void average_plane_avx2(BYTE *p1, const BYTE *p2, int p1_pitch, int p2_pitch, int rowsize, int height) {
  // width is RowSize here
  int mod32_width = rowsize / 32 * 32;
  int mod16_width = rowsize / 16 * 16;

  for(int y = 0; y < height; y++) {
    for(int x = 0; x < mod32_width; x+=32) {
#ifdef AVX2BUG_WORKAROUND
      __m256i src1 = _mm256_load_si256(reinterpret_cast<__m256i*>(p1 + x));
#else
      __m256i src1  = _mm256_stream_load_si256(reinterpret_cast<__m256i*>(p1+x));
#endif
      /*
      00070	8d 40 20	 lea	 eax, DWORD PTR [eax+32]

      ; 70   :       __m256i src1  = _mm256_stream_load_si256(reinterpret_cast<__m256i*>(p1+x));

      00073	c5 fe 6f 40 e0	 vmovdqu ymm0, YMMWORD PTR [eax-32]
      00078	c4 e2 7d 2a c8	 vmovntdqa ymm1, ymm0            ****CRASH HERE! ILLEGAL INSTRUCTION VS15.5.1!!!****

      ; 71   :       __m256i src2  = _mm256_stream_load_si256(const_cast<__m256i*>(reinterpret_cast<const __m256i*>(p2+x)));

      0007d	c5 fe 6f 44 02 e0		 vmovdqu ymm0, YMMWORD PTR [edx+eax-32]
      00083	c4 e2 7d 2a c0	 vmovntdqa ymm0, ymm0

      */
#ifdef AVX2BUG_WORKAROUND
      __m256i src2  = _mm256_load_si256(const_cast<__m256i*>(reinterpret_cast<const __m256i*>(p2+x)));
      #else
      __m256i src2 = _mm256_stream_load_si256(const_cast<__m256i*>(reinterpret_cast<const __m256i*>(p2 + x)));
      #endif
      __m256i dst;
      if constexpr(sizeof(pixel_t) == 1)
        dst  = _mm256_avg_epu8(src1, src2); // 16 pixels
      else // pixel_size == 2
        dst = _mm256_avg_epu16(src1, src2); // 8 pixels

      _mm256_store_si256(reinterpret_cast<__m256i*>(p1+x), dst);
    }

    for(int x = mod32_width; x < mod16_width; x+=16) {
#ifdef AVX2BUG_WORKAROUND
      __m128i src1 = _mm_load_si128(reinterpret_cast<__m128i*>(p1 + x));
      __m128i src2 = _mm_load_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(p2 + x)));
#else
      __m128i src1  = _mm_stream_load_si128(reinterpret_cast<__m128i*>(p1+x));
      __m128i src2  = _mm_stream_load_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(p2+x)));
#endif
      __m128i dst;
      if constexpr(sizeof(pixel_t) == 1)
        dst  = _mm_avg_epu8(src1, src2); // 8 pixels
      else // pixel_size == 2
        dst = _mm_avg_epu16(src1, src2); // 4 pixels

      _mm_store_si128(reinterpret_cast<__m128i*>(p1+x), dst);
    }

    if (mod16_width != rowsize) {
      for (size_t x = mod16_width / sizeof(pixel_t); x < rowsize/sizeof(pixel_t); ++x) {
        reinterpret_cast<pixel_t *>(p1)[x] = (int(reinterpret_cast<pixel_t *>(p1)[x]) + reinterpret_cast<const pixel_t *>(p2)[x] + 1) >> 1;
      }
    }
    p1 += p1_pitch;
    p2 += p2_pitch;
  }
}

// instantiate to let them access from other modules
template void average_plane_avx2<uint8_t>(BYTE *p1, const BYTE *p2, int p1_pitch, int p2_pitch, int rowsize, int height);
template void average_plane_avx2<uint16_t>(BYTE *p1, const BYTE *p2, int p1_pitch, int p2_pitch, int rowsize, int height);


// weighted_merge_planar AVX2 implementations moved to overlay/intel/blend_common_avx2.cpp
// (see weighted_merge_avx2 / weighted_merge_float_avx2)
