// masked_rowprep_avx2.cpp
// AVX2 rowprep implementations + explicit template instantiations.
// Compiled with -mavx2 -mfma (GCC/Clang) or /arch:AVX2 (MSVC) via handle_arch_flags(AVX2).
//
// simd_magic_div_32_avx2 lives in masked_rowprep_avx2.h (inline, needed by merge impl).
// avx2_pack_* helpers are static — internal to this TU only.

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <immintrin.h>
#endif

#include "masked_rowprep_avx2.h"   // own declarations + simd_magic_div_32_avx2 inline
#include <vector>
#include <cstdint>

// ---------------------------------------------------------------------------
// Pack helpers — internal to this TU.
// ---------------------------------------------------------------------------
static AVS_FORCEINLINE __m128i avx2_pack_u16_to_u8(const __m256i& v) {
  return _mm_packus_epi16(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
}
static AVS_FORCEINLINE __m128i avx2_pack_u32_to_u16(const __m256i& v) {
  __m256i packed = _mm256_packus_epi32(v, _mm256_setzero_si256());
  return _mm_unpacklo_epi64(_mm256_castsi256_si128(packed), _mm256_extracti128_si256(packed, 1));
}

// ---------------------------------------------------------------------------
// MASK422 — horizontal 2-tap average.
//   avg[x] = (src[x*2] + src[x*2+1] + 1) >> 1
// ---------------------------------------------------------------------------
template<typename pixel_t, bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2")))
#endif
static void fill_mask422_avx2(
  pixel_t* dst, const pixel_t* src, int width,
  int opacity_i, int half, MagicDiv magic)
{
  int x = 0;
  if constexpr (sizeof(pixel_t) == 1) {
    [[maybe_unused]] const __m256i v_opacity = _mm256_set1_epi16((short)opacity_i);
    [[maybe_unused]] const __m256i v_half16  = _mm256_set1_epi16((short)half);
    [[maybe_unused]] const __m256i v_mdiv    = _mm256_set1_epi16((short)magic.div);
    const __m256i mask_lo = _mm256_set1_epi16(0x00FF);
    for (; x <= width - 16; x += 16) {
      __m256i v    = _mm256_loadu_si256((const __m256i*)(src + x * 2));
      __m256i even = _mm256_and_si256(v, mask_lo);
      __m256i odd  = _mm256_srli_epi16(v, 8);
      __m256i avg  = _mm256_srli_epi16(
        _mm256_add_epi16(_mm256_add_epi16(even, odd), _mm256_set1_epi16(1)), 1);
      if constexpr (!full_opacity) {
        __m256i scaled = _mm256_add_epi16(_mm256_mullo_epi16(avg, v_opacity), v_half16);
        avg = _mm256_srli_epi16(_mm256_mulhi_epu16(scaled, v_mdiv), magic.shift);
      }
      _mm_storeu_si128((__m128i*)(dst + x), avx2_pack_u16_to_u8(avg));
    }
  } else {
    const __m256i ones = _mm256_set1_epi16(1);
    [[maybe_unused]] const __m256i v_opacity32 = _mm256_set1_epi32(opacity_i);
    [[maybe_unused]] const __m256i v_half32    = _mm256_set1_epi32(half);
    for (; x <= width - 8; x += 8) {
      __m256i v     = _mm256_loadu_si256((const __m256i*)(src + x * 2));
      __m256i avg32 = _mm256_srli_epi32(
        _mm256_add_epi32(_mm256_madd_epi16(v, ones), _mm256_set1_epi32(1)), 1);
      if constexpr (!full_opacity)
        avg32 = simd_magic_div_32_avx2(
          _mm256_add_epi32(_mm256_mullo_epi32(avg32, v_opacity32), v_half32),
          magic.div, magic.shift);
      _mm_storeu_si128((__m128i*)(dst + x), avx2_pack_u32_to_u16(avg32));
    }
  }
  for (; x < width; x++) {
    const int avg = (src[x * 2] + src[x * 2 + 1] + 1) >> 1;
    dst[x] = full_opacity ? (pixel_t)avg
           : (pixel_t)magic_div_rt<pixel_t>((uint32_t)avg * (uint32_t)opacity_i + (uint32_t)half, magic);
  }
}

// ---------------------------------------------------------------------------
// MASK422_MPEG2 — horizontal 3-tap triangle filter with sliding window carry.
//   avg[x] = (left + 2*src[x*2] + src[x*2+1] + 2) >> 2
// ---------------------------------------------------------------------------
template<typename pixel_t, bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2")))
#endif
static void fill_mask422_mpeg2_avx2(
  pixel_t* dst, const pixel_t* src, int width,
  int opacity_i, int half, MagicDiv magic)
{
  int x = 0;
  if constexpr (sizeof(pixel_t) == 1) {
    [[maybe_unused]] const __m256i v_opacity = _mm256_set1_epi16((short)opacity_i);
    [[maybe_unused]] const __m256i v_half16  = _mm256_set1_epi16((short)half);
    [[maybe_unused]] const __m256i v_mdiv    = _mm256_set1_epi16((short)magic.div);
    const __m256i mask_lo = _mm256_set1_epi16(0x00FF);
    __m256i prev_carry = _mm256_castsi128_si256(
      _mm_insert_epi16(_mm_setzero_si128(), src[0], 7));

    for (; x <= width - 16; x += 16) {
      __m256i v    = _mm256_loadu_si256((const __m256i*)(src + x * 2));
      __m256i even = _mm256_and_si256(v, mask_lo);
      __m256i odd  = _mm256_srli_epi16(v, 8);

      __m256i shifted_odd = _mm256_permute2x128_si256(odd, prev_carry, 0x02);
      __m256i left = _mm256_alignr_epi8(odd, shifted_odd, 14);

      __m256i res = _mm256_srli_epi16(
        _mm256_add_epi16(
          _mm256_add_epi16(_mm256_add_epi16(left, _mm256_slli_epi16(even, 1)), odd),
          _mm256_set1_epi16(2)), 2);

      if constexpr (!full_opacity) {
        __m256i scaled = _mm256_add_epi16(_mm256_mullo_epi16(res, v_opacity), v_half16);
        res = _mm256_srli_epi16(_mm256_mulhi_epu16(scaled, v_mdiv), magic.shift);
      }
      _mm_storeu_si128((__m128i*)(dst + x), avx2_pack_u16_to_u8(res));

      prev_carry = _mm256_castsi128_si256(
        _mm_insert_epi16(_mm_setzero_si128(),
          _mm_extract_epi16(_mm256_extracti128_si256(odd, 1), 7), 7));
    }
    int right_val = _mm_extract_epi16(_mm256_castsi256_si128(prev_carry), 7);
    for (; x < width; x++) {
      const int left = right_val;
      const int mid  = src[x * 2];
      right_val      = src[x * 2 + 1];
      const int avg  = (left + 2 * mid + right_val + 2) >> 2;
      dst[x] = full_opacity ? (pixel_t)avg
             : (pixel_t)magic_div_rt<pixel_t>((uint32_t)avg * (uint32_t)opacity_i + (uint32_t)half, magic);
    }
  } else {
    [[maybe_unused]] const __m256i v_opacity32 = _mm256_set1_epi32(opacity_i);
    [[maybe_unused]] const __m256i v_half32    = _mm256_set1_epi32(half);
    __m256i prev_carry = _mm256_castsi128_si256(
      _mm_insert_epi32(_mm_setzero_si128(), src[0], 3));

    for (; x <= width - 8; x += 8) {
      __m256i v = _mm256_loadu_si256((const __m256i*)(src + x * 2));
      __m256i lo32 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v));
      __m256i hi32 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(v, 1));

      __m256i sh_le = _mm256_shuffle_epi32(lo32, 0x88);
      __m256i sh_he = _mm256_shuffle_epi32(hi32, 0x88);
      __m256i even32 = _mm256_permute4x64_epi64(
        _mm256_unpacklo_epi64(sh_le, sh_he), _MM_SHUFFLE(3, 1, 2, 0));

      __m256i sh_lo = _mm256_shuffle_epi32(lo32, 0xDD);
      __m256i sh_ho = _mm256_shuffle_epi32(hi32, 0xDD);
      __m256i odd32 = _mm256_permute4x64_epi64(
        _mm256_unpacklo_epi64(sh_lo, sh_ho), _MM_SHUFFLE(3, 1, 2, 0));

      __m256i shifted_odd32 = _mm256_permute2x128_si256(odd32, prev_carry, 0x02);
      __m256i left = _mm256_alignr_epi8(odd32, shifted_odd32, 12);

      __m256i res = _mm256_srli_epi32(
        _mm256_add_epi32(
          _mm256_add_epi32(_mm256_add_epi32(left, _mm256_slli_epi32(even32, 1)), odd32),
          _mm256_set1_epi32(2)), 2);

      if constexpr (!full_opacity)
        res = simd_magic_div_32_avx2(
          _mm256_add_epi32(_mm256_mullo_epi32(res, v_opacity32), v_half32),
          magic.div, magic.shift);

      _mm_storeu_si128((__m128i*)(dst + x), avx2_pack_u32_to_u16(res));

      prev_carry = _mm256_castsi128_si256(
        _mm_insert_epi32(_mm_setzero_si128(),
          _mm_extract_epi32(_mm256_extracti128_si256(odd32, 1), 3), 3));
    }
    int right_val = _mm_extract_epi32(_mm256_castsi256_si128(prev_carry), 3);
    for (; x < width; x++) {
      const int left = right_val;
      const int mid  = src[x * 2];
      right_val      = src[x * 2 + 1];
      const int avg  = (left + 2 * mid + right_val + 2) >> 2;
      dst[x] = full_opacity ? (pixel_t)avg
             : (pixel_t)magic_div_rt<pixel_t>((uint32_t)avg * (uint32_t)opacity_i + (uint32_t)half, magic);
    }
  }
}

// ---------------------------------------------------------------------------
// MASK420 — 2x2 box average (MPEG-1 placement).
//   avg[x] = (row0[x*2]+row0[x*2+1]+row1[x*2]+row1[x*2+1]+2) >> 2
// ---------------------------------------------------------------------------
template<typename pixel_t, bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2")))
#endif
static void fill_mask420_avx2(
  pixel_t* dst, const pixel_t* row0, int mask_pitch, int width,
  int opacity_i, int half, MagicDiv magic)
{
  const pixel_t* row1 = row0 + mask_pitch;
  int x = 0;
  if constexpr (sizeof(pixel_t) == 1) {
    [[maybe_unused]] const __m256i v_opacity = _mm256_set1_epi16((short)opacity_i);
    [[maybe_unused]] const __m256i v_half16  = _mm256_set1_epi16((short)half);
    [[maybe_unused]] const __m256i v_mdiv    = _mm256_set1_epi16((short)magic.div);
    const __m256i mask_lo = _mm256_set1_epi16(0x00FF);
    for (; x <= width - 16; x += 16) {
      __m256i r0 = _mm256_loadu_si256((const __m256i*)(row0 + x * 2));
      __m256i r1 = _mm256_loadu_si256((const __m256i*)(row1 + x * 2));
      __m256i e0 = _mm256_and_si256(r0, mask_lo);
      __m256i o0 = _mm256_srli_epi16(r0, 8);
      __m256i e1 = _mm256_and_si256(r1, mask_lo);
      __m256i o1 = _mm256_srli_epi16(r1, 8);
      __m256i avg = _mm256_srli_epi16(
        _mm256_add_epi16(
          _mm256_add_epi16(_mm256_add_epi16(e0, o0), _mm256_add_epi16(e1, o1)),
          _mm256_set1_epi16(2)), 2);
      if constexpr (!full_opacity) {
        __m256i scaled = _mm256_add_epi16(_mm256_mullo_epi16(avg, v_opacity), v_half16);
        avg = _mm256_srli_epi16(_mm256_mulhi_epu16(scaled, v_mdiv), magic.shift);
      }
      _mm_storeu_si128((__m128i*)(dst + x), avx2_pack_u16_to_u8(avg));
    }
  } else {
    const __m256i ones = _mm256_set1_epi16(1);
    [[maybe_unused]] const __m256i v_opacity32 = _mm256_set1_epi32(opacity_i);
    [[maybe_unused]] const __m256i v_half32    = _mm256_set1_epi32(half);
    for (; x <= width - 8; x += 8) {
      __m256i v0    = _mm256_loadu_si256((const __m256i*)(row0 + x * 2));
      __m256i v1    = _mm256_loadu_si256((const __m256i*)(row1 + x * 2));
      __m256i avg32 = _mm256_srli_epi32(
        _mm256_add_epi32(
          _mm256_add_epi32(_mm256_madd_epi16(v0, ones), _mm256_madd_epi16(v1, ones)),
          _mm256_set1_epi32(2)), 2);
      if constexpr (!full_opacity)
        avg32 = simd_magic_div_32_avx2(
          _mm256_add_epi32(_mm256_mullo_epi32(avg32, v_opacity32), v_half32),
          magic.div, magic.shift);
      _mm_storeu_si128((__m128i*)(dst + x), avx2_pack_u32_to_u16(avg32));
    }
  }
  for (; x < width; x++) {
    const int avg = ((int)row0[x*2] + row0[x*2+1] + row1[x*2] + row1[x*2+1] + 2) >> 2;
    dst[x] = full_opacity ? (pixel_t)avg
           : (pixel_t)magic_div_rt<pixel_t>((uint32_t)avg * (uint32_t)opacity_i + (uint32_t)half, magic);
  }
}

// ---------------------------------------------------------------------------
// MASK420_MPEG2 — horizontal 3-tap triangle filter with vertical 2-row sum and
// sliding-window carry.  Filter:
//   pe[x] = row0[x*2]   + row1[x*2]     (vertical sum of even-indexed pairs)
//   po[x] = row0[x*2+1] + row1[x*2+1]   (vertical sum of odd-indexed pairs)
//   avg[x] = (po[x-1] + 2*pe[x] + po[x] + 4) >> 3
// uint8_t:  16 output pixels / iteration (32-byte load per row → pe/po as uint16)
// uint16_t:  8 output pixels / iteration (32-byte load per row → pe/po as uint32)
// Cross-lane carry: 1 element = 14-byte alignr (uint8_t), 12-byte (uint16_t).
// ---------------------------------------------------------------------------
template<typename pixel_t, bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2")))
#endif
static void fill_mask420_mpeg2_avx2(
  pixel_t* dst, const pixel_t* row0, int mask_pitch, int width,
  int opacity_i, int half, MagicDiv magic)
{
  const pixel_t* row1 = row0 + mask_pitch;
  int x = 0;
  if constexpr (sizeof(pixel_t) == 1) {
    [[maybe_unused]] const __m256i v_opacity = _mm256_set1_epi16((short)opacity_i);
    [[maybe_unused]] const __m256i v_half16  = _mm256_set1_epi16((short)half);
    [[maybe_unused]] const __m256i v_mdiv    = _mm256_set1_epi16((short)magic.div);
    const __m256i mask_lo = _mm256_set1_epi16(0x00FF);
    const int p0 = (int)row0[0] + row1[0];
    __m256i prev_carry = _mm256_castsi128_si256(
      _mm_insert_epi16(_mm_setzero_si128(), p0, 7));

    for (; x <= width - 16; x += 16) {
      __m256i r0 = _mm256_loadu_si256((const __m256i*)(row0 + x * 2));
      __m256i r1 = _mm256_loadu_si256((const __m256i*)(row1 + x * 2));
      __m256i pe = _mm256_add_epi16(_mm256_and_si256(r0, mask_lo),
                                     _mm256_and_si256(r1, mask_lo));
      __m256i po = _mm256_add_epi16(_mm256_srli_epi16(r0, 8),
                                     _mm256_srli_epi16(r1, 8));

      __m256i shifted_odd = _mm256_permute2x128_si256(po, prev_carry, 0x02);
      __m256i left = _mm256_alignr_epi8(po, shifted_odd, 14);

      __m256i res = _mm256_srli_epi16(
        _mm256_add_epi16(
          _mm256_add_epi16(_mm256_add_epi16(left, _mm256_slli_epi16(pe, 1)), po),
          _mm256_set1_epi16(4)), 3);

      if constexpr (!full_opacity) {
        __m256i scaled = _mm256_add_epi16(_mm256_mullo_epi16(res, v_opacity), v_half16);
        res = _mm256_srli_epi16(_mm256_mulhi_epu16(scaled, v_mdiv), magic.shift);
      }
      _mm_storeu_si128((__m128i*)(dst + x), avx2_pack_u16_to_u8(res));

      prev_carry = _mm256_castsi128_si256(
        _mm_insert_epi16(_mm_setzero_si128(),
          _mm_extract_epi16(_mm256_extracti128_si256(po, 1), 7), 7));
    }
    int right_val = _mm_extract_epi16(_mm256_castsi256_si128(prev_carry), 7);
    for (; x < width; x++) {
      const int left = right_val;
      const int mid  = (int)row0[x*2]   + row1[x*2];
      right_val      = (int)row0[x*2+1] + row1[x*2+1];
      const int avg  = (left + 2 * mid + right_val + 4) >> 3;
      dst[x] = full_opacity ? (pixel_t)avg
             : (pixel_t)magic_div_rt<pixel_t>((uint32_t)avg * (uint32_t)opacity_i + (uint32_t)half, magic);
    }
  } else {
    [[maybe_unused]] const __m256i v_opacity32 = _mm256_set1_epi32(opacity_i);
    [[maybe_unused]] const __m256i v_half32    = _mm256_set1_epi32(half);
    const int p0 = (int)row0[0] + row1[0];
    __m256i prev_carry = _mm256_castsi128_si256(
      _mm_insert_epi32(_mm_setzero_si128(), p0, 3));

    for (; x <= width - 8; x += 8) {
      __m256i r0 = _mm256_loadu_si256((const __m256i*)(row0 + x * 2));
      __m256i r1 = _mm256_loadu_si256((const __m256i*)(row1 + x * 2));
      // Expand each 8-uint16 half to 8 uint32 and sum vertically.
      __m256i plo32 = _mm256_add_epi32(
        _mm256_cvtepu16_epi32(_mm256_castsi256_si128(r0)),
        _mm256_cvtepu16_epi32(_mm256_castsi256_si128(r1)));
      __m256i phi32 = _mm256_add_epi32(
        _mm256_cvtepu16_epi32(_mm256_extracti128_si256(r0, 1)),
        _mm256_cvtepu16_epi32(_mm256_extracti128_si256(r1, 1)));

      // Deinterleave even/odd (same as fill_mask422_mpeg2_avx2 uint16_t path).
      __m256i sh_le  = _mm256_shuffle_epi32(plo32, 0x88);
      __m256i sh_he  = _mm256_shuffle_epi32(phi32, 0x88);
      __m256i even32 = _mm256_permute4x64_epi64(
        _mm256_unpacklo_epi64(sh_le, sh_he), _MM_SHUFFLE(3, 1, 2, 0));

      __m256i sh_lo = _mm256_shuffle_epi32(plo32, 0xDD);
      __m256i sh_ho = _mm256_shuffle_epi32(phi32, 0xDD);
      __m256i odd32 = _mm256_permute4x64_epi64(
        _mm256_unpacklo_epi64(sh_lo, sh_ho), _MM_SHUFFLE(3, 1, 2, 0));

      __m256i shifted_odd32 = _mm256_permute2x128_si256(odd32, prev_carry, 0x02);
      __m256i left = _mm256_alignr_epi8(odd32, shifted_odd32, 12);

      __m256i res = _mm256_srli_epi32(
        _mm256_add_epi32(
          _mm256_add_epi32(_mm256_add_epi32(left, _mm256_slli_epi32(even32, 1)), odd32),
          _mm256_set1_epi32(4)), 3);

      if constexpr (!full_opacity)
        res = simd_magic_div_32_avx2(
          _mm256_add_epi32(_mm256_mullo_epi32(res, v_opacity32), v_half32),
          magic.div, magic.shift);

      _mm_storeu_si128((__m128i*)(dst + x), avx2_pack_u32_to_u16(res));

      prev_carry = _mm256_castsi128_si256(
        _mm_insert_epi32(_mm_setzero_si128(),
          _mm_extract_epi32(_mm256_extracti128_si256(odd32, 1), 3), 3));
    }
    int right_val = _mm_extract_epi32(_mm256_castsi256_si128(prev_carry), 3);
    for (; x < width; x++) {
      const int left = right_val;
      const int mid  = (int)row0[x*2]   + row1[x*2];
      right_val      = (int)row0[x*2+1] + row1[x*2+1];
      const int avg  = (left + 2 * mid + right_val + 4) >> 3;
      dst[x] = full_opacity ? (pixel_t)avg
             : (pixel_t)magic_div_rt<pixel_t>((uint32_t)avg * (uint32_t)opacity_i + (uint32_t)half, magic);
    }
  }
}

// ---------------------------------------------------------------------------
// MASK422_TOPLEFT — left co-sited point sample (no averaging).
//   dst[x] = src[x*2]
// ---------------------------------------------------------------------------
template<typename pixel_t, bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2")))
#endif
static void fill_mask422_topleft_avx2(
  pixel_t* dst, const pixel_t* src, int width,
  int opacity_i, int half, MagicDiv magic)
{
  int x = 0;
  if constexpr (sizeof(pixel_t) == 1) {
    [[maybe_unused]] const __m256i v_opacity = _mm256_set1_epi16((short)opacity_i);
    [[maybe_unused]] const __m256i v_half16  = _mm256_set1_epi16((short)half);
    [[maybe_unused]] const __m256i v_mdiv    = _mm256_set1_epi16((short)magic.div);
    const __m256i mask_lo = _mm256_set1_epi16(0x00FF);
    for (; x <= width - 16; x += 16) {
      __m256i v    = _mm256_loadu_si256((const __m256i*)(src + x * 2));
      __m256i even = _mm256_and_si256(v, mask_lo); // left (even) bytes as uint16
      if constexpr (!full_opacity) {
        __m256i scaled = _mm256_add_epi16(_mm256_mullo_epi16(even, v_opacity), v_half16);
        even = _mm256_srli_epi16(_mm256_mulhi_epu16(scaled, v_mdiv), magic.shift);
      }
      _mm_storeu_si128((__m128i*)(dst + x), avx2_pack_u16_to_u8(even));
    }
  } else {
    // 16-bit: grab even-indexed elements (src[x*2], src[x*2+2], ...)
    [[maybe_unused]] const __m256i v_opacity32 = _mm256_set1_epi32(opacity_i);
    [[maybe_unused]] const __m256i v_half32    = _mm256_set1_epi32(half);
    for (; x <= width - 8; x += 8) {
      __m256i v = _mm256_loadu_si256((const __m256i*)(src + x * 2));
      // Deinterleave: keep even-indexed uint16 elements
      __m256i lo32 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v));
      __m256i hi32 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(v, 1));
      __m256i sh_le  = _mm256_shuffle_epi32(lo32, 0x88); // [v0,v2,_,_] in lo64 per lane
      __m256i sh_he  = _mm256_shuffle_epi32(hi32, 0x88);
      __m256i even32 = _mm256_permute4x64_epi64(
        _mm256_unpacklo_epi64(sh_le, sh_he), _MM_SHUFFLE(3, 1, 2, 0));
      if constexpr (!full_opacity)
        even32 = simd_magic_div_32_avx2(
          _mm256_add_epi32(_mm256_mullo_epi32(even32, v_opacity32), v_half32),
          magic.div, magic.shift);
      _mm_storeu_si128((__m128i*)(dst + x), avx2_pack_u32_to_u16(even32));
    }
  }
  for (; x < width; x++) {
    const int val = src[x * 2];
    dst[x] = full_opacity ? (pixel_t)val
           : (pixel_t)magic_div_rt<pixel_t>((uint32_t)val * (uint32_t)opacity_i + (uint32_t)half, magic);
  }
}

// ---------------------------------------------------------------------------
// MASK420_TOPLEFT — top-left co-sited point sample (top row only, no averaging).
//   dst[x] = row0[x*2]
// ---------------------------------------------------------------------------
template<typename pixel_t, bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2")))
#endif
static void fill_mask420_topleft_avx2(
  pixel_t* dst, const pixel_t* row0, int /*mask_pitch*/, int width,
  int opacity_i, int half, MagicDiv magic)
{
  // Identical to fill_mask422_topleft_avx2: top row only, left co-sited.
  fill_mask422_topleft_avx2<pixel_t, full_opacity>(dst, row0, width, opacity_i, half, magic);
}

// ---------------------------------------------------------------------------
// MASK411 — horizontal 4-tap box average.
//   avg[x] = (src[x*4]+src[x*4+1]+src[x*4+2]+src[x*4+3]+2) >> 2
// ---------------------------------------------------------------------------
template<typename pixel_t, bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2")))
#endif
static void fill_mask411_avx2(
  pixel_t* dst, const pixel_t* src, int width,
  int opacity_i, int half, MagicDiv magic)
{
  int x = 0;
  if constexpr (sizeof(pixel_t) == 1) {
    const __m256i zero = _mm256_setzero_si256();
    [[maybe_unused]] const __m256i v_opacity = _mm256_set1_epi16((short)opacity_i);
    [[maybe_unused]] const __m256i v_half16  = _mm256_set1_epi16((short)half);
    [[maybe_unused]] const __m256i v_mdiv    = _mm256_set1_epi16((short)magic.div);
    for (; x <= width - 16; x += 16) {
      __m256i v0 = _mm256_loadu_si256((const __m256i*)(src + x * 4));
      __m256i v1 = _mm256_loadu_si256((const __m256i*)(src + x * 4 + 32));
      __m256i p0 = _mm256_hadd_epi16(
        _mm256_unpacklo_epi8(v0, zero), _mm256_unpackhi_epi8(v0, zero));
      __m256i p1 = _mm256_hadd_epi16(
        _mm256_unpacklo_epi8(v1, zero), _mm256_unpackhi_epi8(v1, zero));
      __m256i avg = _mm256_srli_epi16(
        _mm256_add_epi16(_mm256_hadd_epi16(p0, p1), _mm256_set1_epi16(2)), 2);
      avg = _mm256_permute4x64_epi64(avg, _MM_SHUFFLE(3, 1, 2, 0));
      if constexpr (!full_opacity) {
        __m256i scaled = _mm256_add_epi16(_mm256_mullo_epi16(avg, v_opacity), v_half16);
        avg = _mm256_srli_epi16(_mm256_mulhi_epu16(scaled, v_mdiv), magic.shift);
      }
      _mm_storeu_si128((__m128i*)(dst + x), avx2_pack_u16_to_u8(avg));
    }
  } else {
    const __m256i ones = _mm256_set1_epi16(1);
    [[maybe_unused]] const __m256i v_opacity32 = _mm256_set1_epi32(opacity_i);
    [[maybe_unused]] const __m256i v_half32    = _mm256_set1_epi32(half);
    for (; x <= width - 8; x += 8) {
      __m256i v0 = _mm256_loadu_si256((const __m256i*)(src + x * 4));
      __m256i v1 = _mm256_loadu_si256((const __m256i*)(src + x * 4 + 16));
      __m256i m0 = _mm256_madd_epi16(v0, ones);
      __m256i m1 = _mm256_madd_epi16(v1, ones);
      __m256i avg32 = _mm256_srli_epi32(
        _mm256_add_epi32(_mm256_hadd_epi32(m0, m1), _mm256_set1_epi32(2)), 2);
      avg32 = _mm256_permute4x64_epi64(avg32, _MM_SHUFFLE(3, 1, 2, 0));
      if constexpr (!full_opacity)
        avg32 = simd_magic_div_32_avx2(
          _mm256_add_epi32(_mm256_mullo_epi32(avg32, v_opacity32), v_half32),
          magic.div, magic.shift);
      _mm_storeu_si128((__m128i*)(dst + x), avx2_pack_u32_to_u16(avg32));
    }
  }
  for (; x < width; x++) {
    const int avg = ((int)src[x*4] + src[x*4+1] + src[x*4+2] + src[x*4+3] + 2) >> 2;
    dst[x] = full_opacity ? (pixel_t)avg
           : (pixel_t)magic_div_rt<pixel_t>((uint32_t)avg * (uint32_t)opacity_i + (uint32_t)half, magic);
  }
}

// ---------------------------------------------------------------------------
// prepare_effective_mask_for_row_avx2
// ---------------------------------------------------------------------------
template<MaskMode maskMode, typename pixel_t, bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2")))
#endif
const pixel_t* prepare_effective_mask_for_row_avx2(
  const pixel_t* maskp,
  int mask_pitch,
  int width,
  std::vector<pixel_t>& buf,
  int opacity_i,
  int half,
  MagicDiv magic)
{
  if constexpr (maskMode == MASK444) {
    if constexpr (full_opacity) {
      return maskp;
    } else {
      pixel_t* dst = buf.data();
      int x = 0;
      if constexpr (sizeof(pixel_t) == 1) {
        const __m256i v_opacity = _mm256_set1_epi16((short)opacity_i);
        const __m256i v_half16  = _mm256_set1_epi16((short)half);
        const __m256i v_mdiv    = _mm256_set1_epi16((short)magic.div);
        for (; x <= width - 16; x += 16) {
          __m256i v16 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(maskp + x)));
          __m256i scaled = _mm256_add_epi16(_mm256_mullo_epi16(v16, v_opacity), v_half16);
          __m256i res    = _mm256_srli_epi16(_mm256_mulhi_epu16(scaled, v_mdiv), magic.shift);
          _mm_storeu_si128((__m128i*)(dst + x), avx2_pack_u16_to_u8(res));
        }
      } else {
        const __m256i v_opacity32 = _mm256_set1_epi32(opacity_i);
        const __m256i v_half32    = _mm256_set1_epi32(half);
        for (; x <= width - 8; x += 8) {
          __m256i v32 = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(maskp + x)));
          __m256i res = simd_magic_div_32_avx2(
            _mm256_add_epi32(_mm256_mullo_epi32(v32, v_opacity32), v_half32),
            magic.div, magic.shift);
          _mm_storeu_si128((__m128i*)(dst + x), avx2_pack_u32_to_u16(res));
        }
      }
      for (; x < width; x++)
        dst[x] = static_cast<pixel_t>(
          magic_div_rt<pixel_t>((uint32_t)maskp[x] * (uint32_t)opacity_i + (uint32_t)half, magic));
      return dst;
    }
  }
  else {
    pixel_t* dst = buf.data();
    if constexpr (maskMode == MASK422)
      fill_mask422_avx2<pixel_t, full_opacity>(dst, maskp, width, opacity_i, half, magic);
    else if constexpr (maskMode == MASK422_MPEG2)
      fill_mask422_mpeg2_avx2<pixel_t, full_opacity>(dst, maskp, width, opacity_i, half, magic);
    else if constexpr (maskMode == MASK422_TOPLEFT)
      fill_mask422_topleft_avx2<pixel_t, full_opacity>(dst, maskp, width, opacity_i, half, magic);
    else if constexpr (maskMode == MASK420)
      fill_mask420_avx2<pixel_t, full_opacity>(dst, maskp, mask_pitch, width, opacity_i, half, magic);
    else if constexpr (maskMode == MASK420_MPEG2)
      fill_mask420_mpeg2_avx2<pixel_t, full_opacity>(dst, maskp, mask_pitch, width, opacity_i, half, magic);
    else if constexpr (maskMode == MASK420_TOPLEFT)
      fill_mask420_topleft_avx2<pixel_t, full_opacity>(dst, maskp, mask_pitch, width, opacity_i, half, magic);
    else if constexpr (maskMode == MASK411)
      fill_mask411_avx2<pixel_t, full_opacity>(dst, maskp, width, opacity_i, half, magic);
    return dst;
  }
}

// ---------------------------------------------------------------------------
// prepare_effective_mask_for_row_level_baked_avx2
// ---------------------------------------------------------------------------
template<MaskMode maskMode, typename pixel_t, bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2")))
#endif
const pixel_t* prepare_effective_mask_for_row_level_baked_avx2(
  const pixel_t* maskp,
  int mask_pitch,
  int width,
  std::vector<pixel_t>& buf,
  int level,
  int bits_per_pixel)
{
  if constexpr (maskMode == MASK444) {
    if constexpr (full_opacity)
      return maskp;
    pixel_t* dst = buf.data();
    int x = 0;
    if constexpr (sizeof(pixel_t) == 1) {
      const __m256i v_level = _mm256_set1_epi16((short)level);
      const __m256i v_one   = _mm256_set1_epi16(1);
      for (; x <= width - 16; x += 16) {
        __m256i v = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(maskp + x)));
        __m256i r = _mm256_srli_epi16(_mm256_add_epi16(_mm256_mullo_epi16(v, v_level), v_one), 8);
        _mm_storeu_si128((__m128i*)(dst + x), avx2_pack_u16_to_u8(r));
      }
    } else {
      const __m256i v_level32 = _mm256_set1_epi32(level);
      const __m256i v_one32   = _mm256_set1_epi32(1);
      const __m128i v_bpp     = _mm_cvtsi32_si128(bits_per_pixel);
      for (; x <= width - 8; x += 8) {
        __m256i v = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(maskp + x)));
        __m256i r = _mm256_srl_epi32(_mm256_add_epi32(_mm256_mullo_epi32(v, v_level32), v_one32), v_bpp);
        _mm_storeu_si128((__m128i*)(dst + x), avx2_pack_u32_to_u16(r));
      }
    }
    for (; x < width; x++)
      dst[x] = static_cast<pixel_t>(((uint32_t)maskp[x] * (uint32_t)level + 1u) >> bits_per_pixel);
    return dst;
  }
  else {
    pixel_t* dst = buf.data();
    // Pass 1: spatial average, Overlay baking suppressed (full_opacity=true to skip it).
    if constexpr (maskMode == MASK422)
      fill_mask422_avx2<pixel_t, true>(dst, maskp, width, 0, 0, {});
    else if constexpr (maskMode == MASK422_MPEG2)
      fill_mask422_mpeg2_avx2<pixel_t, true>(dst, maskp, width, 0, 0, {});
    else if constexpr (maskMode == MASK422_TOPLEFT)
      fill_mask422_topleft_avx2<pixel_t, true>(dst, maskp, width, 0, 0, {});
    else if constexpr (maskMode == MASK420)
      fill_mask420_avx2<pixel_t, true>(dst, maskp, mask_pitch, width, 0, 0, {});
    else if constexpr (maskMode == MASK420_MPEG2)
      fill_mask420_mpeg2_avx2<pixel_t, true>(dst, maskp, mask_pitch, width, 0, 0, {});
    else if constexpr (maskMode == MASK420_TOPLEFT)
      fill_mask420_topleft_avx2<pixel_t, true>(dst, maskp, mask_pitch, width, 0, 0, {});
    else if constexpr (maskMode == MASK411)
      fill_mask411_avx2<pixel_t, true>(dst, maskp, width, 0, 0, {});
    // Pass 2: level baking (only when !full_opacity).
    if constexpr (!full_opacity) {
      int x = 0;
      if constexpr (sizeof(pixel_t) == 1) {
        const __m256i v_level = _mm256_set1_epi16((short)level);
        const __m256i v_one   = _mm256_set1_epi16(1);
        for (; x <= width - 16; x += 16) {
          __m256i v = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(dst + x)));
          __m256i r = _mm256_srli_epi16(_mm256_add_epi16(_mm256_mullo_epi16(v, v_level), v_one), 8);
          _mm_storeu_si128((__m128i*)(dst + x), avx2_pack_u16_to_u8(r));
        }
      } else {
        const __m256i v_level32 = _mm256_set1_epi32(level);
        const __m256i v_one32   = _mm256_set1_epi32(1);
        const __m128i v_bpp     = _mm_cvtsi32_si128(bits_per_pixel);
        for (; x <= width - 8; x += 8) {
          __m256i v = _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)(dst + x)));
          __m256i r = _mm256_srl_epi32(_mm256_add_epi32(_mm256_mullo_epi32(v, v_level32), v_one32), v_bpp);
          _mm_storeu_si128((__m128i*)(dst + x), avx2_pack_u32_to_u16(r));
        }
      }
      for (; x < width; x++)
        dst[x] = static_cast<pixel_t>(((uint32_t)dst[x] * (uint32_t)level + 1u) >> bits_per_pixel);
    }
    return dst;
  }
}

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------

// prepare_effective_mask_for_row_avx2
#define INST_PREP_AVX2(mm, pt) \
  template const pt* prepare_effective_mask_for_row_avx2<mm, pt, true> (const pt*, int, int, std::vector<pt>&, int, int, MagicDiv); \
  template const pt* prepare_effective_mask_for_row_avx2<mm, pt, false>(const pt*, int, int, std::vector<pt>&, int, int, MagicDiv);
INST_PREP_AVX2(MASK444,          uint8_t)   INST_PREP_AVX2(MASK444,          uint16_t)
INST_PREP_AVX2(MASK420,          uint8_t)   INST_PREP_AVX2(MASK420,          uint16_t)
INST_PREP_AVX2(MASK420_MPEG2,    uint8_t)   INST_PREP_AVX2(MASK420_MPEG2,    uint16_t)
INST_PREP_AVX2(MASK420_TOPLEFT,  uint8_t)   INST_PREP_AVX2(MASK420_TOPLEFT,  uint16_t)
INST_PREP_AVX2(MASK422,          uint8_t)   INST_PREP_AVX2(MASK422,          uint16_t)
INST_PREP_AVX2(MASK422_MPEG2,    uint8_t)   INST_PREP_AVX2(MASK422_MPEG2,    uint16_t)
INST_PREP_AVX2(MASK422_TOPLEFT,  uint8_t)   INST_PREP_AVX2(MASK422_TOPLEFT,  uint16_t)
INST_PREP_AVX2(MASK411,          uint8_t)   INST_PREP_AVX2(MASK411,          uint16_t)
#undef INST_PREP_AVX2

// prepare_effective_mask_for_row_level_baked_avx2
#define INST_BAKED_AVX2(mm, pt) \
  template const pt* prepare_effective_mask_for_row_level_baked_avx2<mm, pt, true> (const pt*, int, int, std::vector<pt>&, int, int); \
  template const pt* prepare_effective_mask_for_row_level_baked_avx2<mm, pt, false>(const pt*, int, int, std::vector<pt>&, int, int);
INST_BAKED_AVX2(MASK444,          uint8_t)   INST_BAKED_AVX2(MASK444,          uint16_t)
INST_BAKED_AVX2(MASK420,          uint8_t)   INST_BAKED_AVX2(MASK420,          uint16_t)
INST_BAKED_AVX2(MASK420_MPEG2,    uint8_t)   INST_BAKED_AVX2(MASK420_MPEG2,    uint16_t)
INST_BAKED_AVX2(MASK420_TOPLEFT,  uint8_t)   INST_BAKED_AVX2(MASK420_TOPLEFT,  uint16_t)
INST_BAKED_AVX2(MASK422,          uint8_t)   INST_BAKED_AVX2(MASK422,          uint16_t)
INST_BAKED_AVX2(MASK422_MPEG2,    uint8_t)   INST_BAKED_AVX2(MASK422_MPEG2,    uint16_t)
INST_BAKED_AVX2(MASK422_TOPLEFT,  uint8_t)   INST_BAKED_AVX2(MASK422_TOPLEFT,  uint16_t)
INST_BAKED_AVX2(MASK411,          uint8_t)   INST_BAKED_AVX2(MASK411,          uint16_t)
#undef INST_BAKED_AVX2
