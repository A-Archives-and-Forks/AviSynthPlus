// masked_rowprep_avx2.h
// AVX2 rowprep function declarations.
// Implementations and explicit instantiations are in masked_rowprep_avx2.cpp.
//
// simd_magic_div_32_avx2 is kept inline here — used inside masked_merge_avx2_impl.hpp
// hot loops and must remain inlineable within those TUs.
//
// masked_rowprep_sse41.h is included for:
//   - simd_magic_div_32 inline (used by masked_merge_sse41_impl.hpp when included
//     in AVX2 TUs)

#pragma once

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <immintrin.h>   // AVX2
#endif

#include "masked_rowprep_sse41.h"   // simd_magic_div_32, fill_mask420_mpeg2_sse41 decl

// ---------------------------------------------------------------------------
// simd_magic_div_32_avx2
// Inline here: used by masked_merge_avx2_impl.hpp in inner blend loops.
// ---------------------------------------------------------------------------
static AVS_FORCEINLINE __m256i simd_magic_div_32_avx2(__m256i val, uint32_t magic, int shift) {
  __m256i v_magic  = _mm256_set1_epi64x(magic);
  __m256i res_even = _mm256_mul_epu32(val, v_magic);
  __m256i res_odd  = _mm256_mul_epu32(_mm256_srli_si256(val, 4), v_magic);
  res_even = _mm256_srli_epi64(res_even, 32 + shift);
  res_odd  = _mm256_srli_epi64(res_odd,  32 + shift);
  __m256i res_odd_shifted = _mm256_slli_epi64(res_odd, 32);
  return _mm256_blend_epi32(res_even, res_odd_shifted, 0xAA);
}

// ---------------------------------------------------------------------------
// prepare_effective_mask_for_row_avx2
// Declared here; defined + explicitly instantiated in masked_rowprep_avx2.cpp.
// ---------------------------------------------------------------------------
template<MaskMode maskMode, typename pixel_t, bool full_opacity = true>
const pixel_t* prepare_effective_mask_for_row_avx2(
  const pixel_t* maskp,
  int mask_pitch,
  int width,
  std::vector<pixel_t>& buf,
  int opacity_i = 0,
  int half = 0,
  MagicDiv magic = {});

// ---------------------------------------------------------------------------
// prepare_effective_mask_for_row_level_baked_avx2
// Layer-style baking: result = (avg * level + 1) >> bits_per_pixel.
// Declared here; defined + explicitly instantiated in masked_rowprep_avx2.cpp.
// ---------------------------------------------------------------------------
template<MaskMode maskMode, typename pixel_t, bool full_opacity = false>
const pixel_t* prepare_effective_mask_for_row_level_baked_avx2(
  const pixel_t* maskp,
  int mask_pitch,
  int width,
  std::vector<pixel_t>& buf,
  int level = 0,
  int bits_per_pixel = 8);
