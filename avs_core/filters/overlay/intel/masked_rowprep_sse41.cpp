// masked_rowprep_sse41.cpp
// SSE4.1 rowprep implementations + explicit template instantiations.
// Compiled with -msse4.1 (GCC/Clang) or /arch:SSE2 (MSVC) via handle_arch_flags(SSE41).
//
// simd_magic_div_32 lives in masked_rowprep_sse41.h (inline, needed by merge impl).
// All fill_mask*_sse41 helpers are static (internal to this TU).

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <smmintrin.h>
#endif

#include "masked_rowprep_sse41.h"   // own declarations + simd_magic_div_32 inline

#include <vector>
#include <cstdint>

// ---------------------------------------------------------------------------
// Internal helper — deinterleave 16 uint8 → two int16 vectors (even/odd bytes).
// Only used within this TU by the fill_mask* functions below.
// ---------------------------------------------------------------------------
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static AVS_FORCEINLINE void deinterleave_u8_epi16(
  __m128i v, __m128i& even_out, __m128i& odd_out)
{
  even_out = _mm_and_si128(v, _mm_set1_epi16(0x00FF));
  odd_out  = _mm_srli_epi16(v, 8);
}

// ---------------------------------------------------------------------------
// MASK422 — horizontal 2-tap average, no inter-row state.
//   avg[x] = (src[x*2] + src[x*2+1] + 1) >> 1
//   dst[x] = full_opacity ? avg : (avg * opacity_i + half) / max
// ---------------------------------------------------------------------------
template<typename pixel_t, bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static void fill_mask422_sse41(
  pixel_t* dst, const pixel_t* src, int width,
  int opacity_i, int half, MagicDiv magic)
{
  int x = 0;
  if constexpr (sizeof(pixel_t) == 1) {
    // 8-bit: 16 luma bytes → 8 chroma bytes per iteration
    [[maybe_unused]] const __m128i v_opacity = _mm_set1_epi16((short)opacity_i);
    [[maybe_unused]] const __m128i v_half16  = _mm_set1_epi16((short)half);
    [[maybe_unused]] const __m128i v_mdiv    = _mm_set1_epi16((short)magic.div);
    for (; x <= width - 8; x += 8) {
      __m128i v = _mm_loadu_si128((const __m128i*)(src + x * 2));
      __m128i even, odd;
      deinterleave_u8_epi16(v, even, odd);
      __m128i avg = _mm_srli_epi16(_mm_add_epi16(_mm_add_epi16(even, odd), _mm_set1_epi16(1)), 1);
      if constexpr (!full_opacity) {
        // (avg * opacity + half) / max — all fit in uint16
        __m128i scaled = _mm_add_epi16(_mm_mullo_epi16(avg, v_opacity), v_half16);
        avg = _mm_srli_epi16(_mm_mulhi_epu16(scaled, v_mdiv), magic.shift);
      }
      _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi16(avg, avg));
    }
  } else {
    // 16-bit: 8 uint16 luma → 4 uint16 chroma per iteration.
    const __m128i ones = _mm_set1_epi16(1);
    [[maybe_unused]] const __m128i v_opacity32 = _mm_set1_epi32(opacity_i);
    [[maybe_unused]] const __m128i v_half32    = _mm_set1_epi32(half);
    for (; x <= width - 4; x += 4) {
      __m128i v     = _mm_loadu_si128((const __m128i*)(src + x * 2));
      __m128i avg32 = _mm_srli_epi32(_mm_add_epi32(_mm_madd_epi16(v, ones), _mm_set1_epi32(1)), 1);
      if constexpr (!full_opacity)
        avg32 = simd_magic_div_32(
          _mm_add_epi32(_mm_mullo_epi32(avg32, v_opacity32), v_half32),
          magic.div, magic.shift);
      _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi32(avg32, avg32));
    }
  }
  for (; x < width; x++) {
    const int avg = (src[x * 2] + src[x * 2 + 1] + 1) >> 1;
    dst[x] = full_opacity ? (pixel_t)avg
           : (pixel_t)magic_div_rt<pixel_t>((uint32_t)avg * (uint32_t)opacity_i + (uint32_t)half, magic);
  }
}

// ---------------------------------------------------------------------------
// MASK422_MPEG2 — horizontal 3-tap triangle filter with sliding window.
//   avg[x] = (left + 2*src[x*2] + src[x*2+1] + 2) >> 2
//   dst[x] = full_opacity ? avg : (avg * opacity_i + half) / max
// ---------------------------------------------------------------------------
template<typename pixel_t, bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static void fill_mask422_mpeg2_sse41(
  pixel_t* dst, const pixel_t* src, int width,
  int opacity_i, int half, MagicDiv magic)
{
  int x = 0;
  if constexpr (sizeof(pixel_t) == 1) {
    [[maybe_unused]] const __m128i v_opacity = _mm_set1_epi16((short)opacity_i);
    [[maybe_unused]] const __m128i v_half16  = _mm_set1_epi16((short)half);
    [[maybe_unused]] const __m128i v_mdiv    = _mm_set1_epi16((short)magic.div);
    __m128i prev_carry = _mm_insert_epi16(_mm_setzero_si128(), src[0], 7);
    for (; x <= width - 8; x += 8) {
      __m128i v = _mm_loadu_si128((const __m128i*)(src + x * 2));
      __m128i even, odd;
      deinterleave_u8_epi16(v, even, odd);
      __m128i left = _mm_alignr_epi8(odd, prev_carry, 14);
      __m128i res  = _mm_srli_epi16(
        _mm_add_epi16(
          _mm_add_epi16(_mm_add_epi16(left, _mm_slli_epi16(even, 1)), odd),
          _mm_set1_epi16(2)), 2);
      if constexpr (!full_opacity) {
        __m128i scaled = _mm_add_epi16(_mm_mullo_epi16(res, v_opacity), v_half16);
        res = _mm_srli_epi16(_mm_mulhi_epu16(scaled, v_mdiv), magic.shift);
      }
      _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi16(res, res));
      prev_carry = _mm_insert_epi16(_mm_setzero_si128(), _mm_extract_epi16(odd, 7), 7);
    }
    int right_val = _mm_extract_epi16(prev_carry, 7);
    for (; x < width; x++) {
      const int left = right_val;
      const int mid  = src[x * 2];
      right_val      = src[x * 2 + 1];
      const int avg  = (left + 2 * mid + right_val + 2) >> 2;
      dst[x] = full_opacity ? (pixel_t)avg
             : (pixel_t)magic_div_rt<pixel_t>((uint32_t)avg * (uint32_t)opacity_i + (uint32_t)half, magic);
    }
  } else {
    [[maybe_unused]] const __m128i v_opacity32 = _mm_set1_epi32(opacity_i);
    [[maybe_unused]] const __m128i v_half32    = _mm_set1_epi32(half);
    __m128i prev_carry = _mm_insert_epi32(_mm_setzero_si128(), src[0], 3);
    for (; x <= width - 4; x += 4) {
      __m128i v = _mm_loadu_si128((const __m128i*)(src + x * 2));
      __m128i lo32  = _mm_cvtepu16_epi32(v);
      __m128i hi32  = _mm_cvtepu16_epi32(_mm_srli_si128(v, 8));
      __m128i even32 = _mm_unpacklo_epi64(
        _mm_shuffle_epi32(lo32, _MM_SHUFFLE(2, 0, 2, 0)),
        _mm_shuffle_epi32(hi32, _MM_SHUFFLE(2, 0, 2, 0)));
      __m128i odd32 = _mm_unpacklo_epi64(
        _mm_shuffle_epi32(lo32, _MM_SHUFFLE(3, 1, 3, 1)),
        _mm_shuffle_epi32(hi32, _MM_SHUFFLE(3, 1, 3, 1)));
      __m128i left = _mm_alignr_epi8(odd32, prev_carry, 12);
      __m128i res  = _mm_srli_epi32(
        _mm_add_epi32(
          _mm_add_epi32(_mm_add_epi32(left, _mm_slli_epi32(even32, 1)), odd32),
          _mm_set1_epi32(2)), 2);
      if constexpr (!full_opacity)
        res = simd_magic_div_32(
          _mm_add_epi32(_mm_mullo_epi32(res, v_opacity32), v_half32),
          magic.div, magic.shift);
      _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi32(res, res));
      prev_carry = _mm_insert_epi32(_mm_setzero_si128(), _mm_extract_epi32(odd32, 3), 3);
    }
    int right_val = _mm_extract_epi32(prev_carry, 3);
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
// MASK420 — 2×2 box average (MPEG-1 placement). No inter-row state.
//   avg[x] = (row0[x*2]+row0[x*2+1]+row1[x*2]+row1[x*2+1]+2) >> 2
//   dst[x] = full_opacity ? avg : (avg * opacity_i + half) / max
// ---------------------------------------------------------------------------
template<typename pixel_t, bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static void fill_mask420_sse41(
  pixel_t* dst, const pixel_t* row0, int mask_pitch, int width,
  int opacity_i, int half, MagicDiv magic)
{
  const pixel_t* row1 = row0 + mask_pitch;
  int x = 0;
  if constexpr (sizeof(pixel_t) == 1) {
    [[maybe_unused]] const __m128i v_opacity = _mm_set1_epi16((short)opacity_i);
    [[maybe_unused]] const __m128i v_half16  = _mm_set1_epi16((short)half);
    [[maybe_unused]] const __m128i v_mdiv    = _mm_set1_epi16((short)magic.div);
    for (; x <= width - 8; x += 8) {
      __m128i r0 = _mm_loadu_si128((const __m128i*)(row0 + x * 2));
      __m128i r1 = _mm_loadu_si128((const __m128i*)(row1 + x * 2));
      __m128i e0, o0, e1, o1;
      deinterleave_u8_epi16(r0, e0, o0);
      deinterleave_u8_epi16(r1, e1, o1);
      __m128i avg = _mm_srli_epi16(
        _mm_add_epi16(_mm_add_epi16(_mm_add_epi16(e0, o0), _mm_add_epi16(e1, o1)), _mm_set1_epi16(2)), 2);
      if constexpr (!full_opacity) {
        __m128i scaled = _mm_add_epi16(_mm_mullo_epi16(avg, v_opacity), v_half16);
        avg = _mm_srli_epi16(_mm_mulhi_epu16(scaled, v_mdiv), magic.shift);
      }
      _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi16(avg, avg));
    }
  } else {
    const __m128i ones = _mm_set1_epi16(1);
    [[maybe_unused]] const __m128i v_opacity32 = _mm_set1_epi32(opacity_i);
    [[maybe_unused]] const __m128i v_half32    = _mm_set1_epi32(half);
    for (; x <= width - 4; x += 4) {
      __m128i v0    = _mm_loadu_si128((const __m128i*)(row0 + x * 2));
      __m128i v1    = _mm_loadu_si128((const __m128i*)(row1 + x * 2));
      __m128i avg32 = _mm_srli_epi32(
        _mm_add_epi32(_mm_add_epi32(_mm_madd_epi16(v0, ones), _mm_madd_epi16(v1, ones)), _mm_set1_epi32(2)), 2);
      if constexpr (!full_opacity)
        avg32 = simd_magic_div_32(
          _mm_add_epi32(_mm_mullo_epi32(avg32, v_opacity32), v_half32),
          magic.div, magic.shift);
      _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi32(avg32, avg32));
    }
  }
  for (; x < width; x++) {
    const int avg = ((int)row0[x*2] + row0[x*2+1] + row1[x*2] + row1[x*2+1] + 2) >> 2;
    dst[x] = full_opacity ? (pixel_t)avg
           : (pixel_t)magic_div_rt<pixel_t>((uint32_t)avg * (uint32_t)opacity_i + (uint32_t)half, magic);
  }
}

// ---------------------------------------------------------------------------
// MASK420_MPEG2 — 2-row vertical sum + horizontal 3-tap triangle filter.
//   avg[x] = (left + 2*P[x*2] + P[x*2+1] + 4) >> 3  where P[k] = row0[k]+row1[k]
//   dst[x] = full_opacity ? avg : (avg * opacity_i + half) / max
// ---------------------------------------------------------------------------
template<typename pixel_t, bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static void fill_mask420_mpeg2_sse41(
  pixel_t* dst, const pixel_t* row0, int mask_pitch, int width,
  int opacity_i, int half, MagicDiv magic)
{
  const pixel_t* row1 = row0 + mask_pitch;
  int x = 0;
  if constexpr (sizeof(pixel_t) == 1) {
    [[maybe_unused]] const __m128i v_opacity = _mm_set1_epi16((short)opacity_i);
    [[maybe_unused]] const __m128i v_half16  = _mm_set1_epi16((short)half);
    [[maybe_unused]] const __m128i v_mdiv    = _mm_set1_epi16((short)magic.div);
    const int p0 = (int)row0[0] + row1[0];
    __m128i prev_carry = _mm_insert_epi16(_mm_setzero_si128(), p0, 7);

    for (; x <= width - 8; x += 8) {
      __m128i r0 = _mm_loadu_si128((const __m128i*)(row0 + x * 2));
      __m128i r1 = _mm_loadu_si128((const __m128i*)(row1 + x * 2));
      __m128i plo = _mm_add_epi16(_mm_unpacklo_epi8(r0, _mm_setzero_si128()),
                                   _mm_unpacklo_epi8(r1, _mm_setzero_si128()));
      __m128i phi = _mm_add_epi16(_mm_unpackhi_epi8(r0, _mm_setzero_si128()),
                                   _mm_unpackhi_epi8(r1, _mm_setzero_si128()));

      const __m128i shuf_even = _mm_setr_epi8(0,1, 4,5, 8,9, 12,13, -1,-1,-1,-1,-1,-1,-1,-1);
      const __m128i shuf_odd  = _mm_setr_epi8(2,3, 6,7, 10,11, 14,15, -1,-1,-1,-1,-1,-1,-1,-1);
      __m128i pe = _mm_unpacklo_epi64(_mm_shuffle_epi8(plo, shuf_even),
                                      _mm_shuffle_epi8(phi, shuf_even));
      __m128i po = _mm_unpacklo_epi64(_mm_shuffle_epi8(plo, shuf_odd),
                                      _mm_shuffle_epi8(phi, shuf_odd));

      __m128i left = _mm_alignr_epi8(po, prev_carry, 14);
      __m128i res  = _mm_srli_epi16(
        _mm_add_epi16(
          _mm_add_epi16(_mm_add_epi16(left, _mm_slli_epi16(pe, 1)), po),
          _mm_set1_epi16(4)), 3);
      if constexpr (!full_opacity) {
        __m128i scaled = _mm_add_epi16(_mm_mullo_epi16(res, v_opacity), v_half16);
        res = _mm_srli_epi16(_mm_mulhi_epu16(scaled, v_mdiv), magic.shift);
      }
      _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi16(res, res));
      prev_carry = _mm_insert_epi16(_mm_setzero_si128(), _mm_extract_epi16(po, 7), 7);
    }
    int right_val = _mm_extract_epi16(prev_carry, 7);
    for (; x < width; x++) {
      const int left = right_val;
      const int mid  = (int)row0[x*2]   + row1[x*2];
      right_val      = (int)row0[x*2+1] + row1[x*2+1];
      const int avg  = (left + 2 * mid + right_val + 4) >> 3;
      dst[x] = full_opacity ? (pixel_t)avg
             : (pixel_t)magic_div_rt<pixel_t>((uint32_t)avg * (uint32_t)opacity_i + (uint32_t)half, magic);
    }
  } else {
    [[maybe_unused]] const __m128i v_opacity32 = _mm_set1_epi32(opacity_i);
    [[maybe_unused]] const __m128i v_half32    = _mm_set1_epi32(half);
    const int p0 = (int)row0[0] + row1[0];
    __m128i prev_carry = _mm_insert_epi32(_mm_setzero_si128(), p0, 3);

    for (; x <= width - 4; x += 4) {
      __m128i v0  = _mm_loadu_si128((const __m128i*)(row0 + x * 2));
      __m128i v1  = _mm_loadu_si128((const __m128i*)(row1 + x * 2));
      __m128i plo = _mm_add_epi32(_mm_cvtepu16_epi32(v0), _mm_cvtepu16_epi32(v1));
      __m128i phi = _mm_add_epi32(_mm_cvtepu16_epi32(_mm_srli_si128(v0, 8)),
                                   _mm_cvtepu16_epi32(_mm_srli_si128(v1, 8)));
      __m128i pe = _mm_unpacklo_epi64(
        _mm_shuffle_epi32(plo, _MM_SHUFFLE(2, 0, 2, 0)),
        _mm_shuffle_epi32(phi, _MM_SHUFFLE(2, 0, 2, 0)));
      __m128i po = _mm_unpacklo_epi64(
        _mm_shuffle_epi32(plo, _MM_SHUFFLE(3, 1, 3, 1)),
        _mm_shuffle_epi32(phi, _MM_SHUFFLE(3, 1, 3, 1)));

      __m128i left = _mm_alignr_epi8(po, prev_carry, 12);
      __m128i res  = _mm_srli_epi32(
        _mm_add_epi32(
          _mm_add_epi32(_mm_add_epi32(left, _mm_slli_epi32(pe, 1)), po),
          _mm_set1_epi32(4)), 3);
      if constexpr (!full_opacity)
        res = simd_magic_div_32(
          _mm_add_epi32(_mm_mullo_epi32(res, v_opacity32), v_half32),
          magic.div, magic.shift);
      _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi32(res, res));
      prev_carry = _mm_insert_epi32(_mm_setzero_si128(), _mm_extract_epi32(po, 3), 3);
    }
    int right_val = _mm_extract_epi32(prev_carry, 3);
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
//   dst[x] = full_opacity ? dst[x] : (dst[x] * opacity_i + half) / max
// ---------------------------------------------------------------------------
template<typename pixel_t, bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static void fill_mask422_topleft_sse41(
  pixel_t* dst, const pixel_t* src, int width,
  int opacity_i, int half, MagicDiv magic)
{
  int x = 0;
  if constexpr (sizeof(pixel_t) == 1) {
    [[maybe_unused]] const __m128i v_opacity = _mm_set1_epi16((short)opacity_i);
    [[maybe_unused]] const __m128i v_half16  = _mm_set1_epi16((short)half);
    [[maybe_unused]] const __m128i v_mdiv    = _mm_set1_epi16((short)magic.div);
    for (; x <= width - 8; x += 8) {
      __m128i v    = _mm_loadu_si128((const __m128i*)(src + x * 2));
      __m128i even = _mm_and_si128(v, _mm_set1_epi16(0x00FF)); // left (even) bytes
      if constexpr (!full_opacity) {
        __m128i scaled = _mm_add_epi16(_mm_mullo_epi16(even, v_opacity), v_half16);
        even = _mm_srli_epi16(_mm_mulhi_epu16(scaled, v_mdiv), magic.shift);
      }
      _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi16(even, even));
    }
  } else {
    // 16-bit: grab even-indexed uint16 elements (src[x*2], src[x*2+2], ...)
    // Load 8 uint16: [a,b,c,d,e,f,g,h] → want [a,c,e,g]
    const __m128i shuf = _mm_setr_epi8(0,1, 4,5, 8,9, 12,13, -1,-1,-1,-1,-1,-1,-1,-1);
    [[maybe_unused]] const __m128i v_opacity32 = _mm_set1_epi32(opacity_i);
    [[maybe_unused]] const __m128i v_half32    = _mm_set1_epi32(half);
    for (; x <= width - 4; x += 4) {
      __m128i v    = _mm_loadu_si128((const __m128i*)(src + x * 2));
      __m128i even = _mm_shuffle_epi8(v, shuf); // [a,c,e,g,0,...] as 4 uint16
      if constexpr (!full_opacity) {
        __m128i even32 = _mm_cvtepu16_epi32(even);
        even32 = simd_magic_div_32(
          _mm_add_epi32(_mm_mullo_epi32(even32, v_opacity32), v_half32),
          magic.div, magic.shift);
        even = _mm_packus_epi32(even32, even32);
      }
      _mm_storel_epi64((__m128i*)(dst + x), even);
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
__attribute__((__target__("sse4.1")))
#endif
static void fill_mask420_topleft_sse41(
  pixel_t* dst, const pixel_t* row0, int /*mask_pitch*/, int width,
  int opacity_i, int half, MagicDiv magic)
{
  // Identical to fill_mask422_topleft_sse41: top row only, left co-sited.
  fill_mask422_topleft_sse41<pixel_t, full_opacity>(dst, row0, width, opacity_i, half, magic);
}

// ---------------------------------------------------------------------------
// MASK411 — horizontal 4-tap box average.
//   avg[x] = (src[x*4]+src[x*4+1]+src[x*4+2]+src[x*4+3]+2) >> 2
//   dst[x] = full_opacity ? avg : (avg * opacity_i + half) / max
// ---------------------------------------------------------------------------
template<typename pixel_t, bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static void fill_mask411_sse41(
  pixel_t* dst, const pixel_t* src, int width,
  int opacity_i, int half, MagicDiv magic)
{
  int x = 0;
  if constexpr (sizeof(pixel_t) == 1) {
    const __m128i zero = _mm_setzero_si128();
    [[maybe_unused]] const __m128i v_opacity = _mm_set1_epi16((short)opacity_i);
    [[maybe_unused]] const __m128i v_half16  = _mm_set1_epi16((short)half);
    [[maybe_unused]] const __m128i v_mdiv    = _mm_set1_epi16((short)magic.div);
    for (; x <= width - 8; x += 8) {
      __m128i v0 = _mm_loadu_si128((const __m128i*)(src + x * 4));
      __m128i v1 = _mm_loadu_si128((const __m128i*)(src + x * 4 + 16));
      __m128i p0 = _mm_hadd_epi16(_mm_unpacklo_epi8(v0, zero), _mm_unpackhi_epi8(v0, zero));
      __m128i p1 = _mm_hadd_epi16(_mm_unpacklo_epi8(v1, zero), _mm_unpackhi_epi8(v1, zero));
      __m128i avg = _mm_srli_epi16(_mm_add_epi16(_mm_hadd_epi16(p0, p1), _mm_set1_epi16(2)), 2);
      if constexpr (!full_opacity) {
        __m128i scaled = _mm_add_epi16(_mm_mullo_epi16(avg, v_opacity), v_half16);
        avg = _mm_srli_epi16(_mm_mulhi_epu16(scaled, v_mdiv), magic.shift);
      }
      _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi16(avg, avg));
    }
  } else {
    const __m128i ones = _mm_set1_epi16(1);
    [[maybe_unused]] const __m128i v_opacity32 = _mm_set1_epi32(opacity_i);
    [[maybe_unused]] const __m128i v_half32    = _mm_set1_epi32(half);
    for (; x <= width - 4; x += 4) {
      __m128i v0    = _mm_loadu_si128((const __m128i*)(src + x * 4));
      __m128i v1    = _mm_loadu_si128((const __m128i*)(src + x * 4 + 8));
      __m128i avg32 = _mm_srli_epi32(
        _mm_add_epi32(_mm_hadd_epi32(_mm_madd_epi16(v0, ones), _mm_madd_epi16(v1, ones)),
                      _mm_set1_epi32(2)), 2);
      if constexpr (!full_opacity)
        avg32 = simd_magic_div_32(
          _mm_add_epi32(_mm_mullo_epi32(avg32, v_opacity32), v_half32),
          magic.div, magic.shift);
      _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi32(avg32, avg32));
    }
  }
  for (; x < width; x++) {
    const int avg = ((int)src[x*4] + src[x*4+1] + src[x*4+2] + src[x*4+3] + 2) >> 2;
    dst[x] = full_opacity ? (pixel_t)avg
           : (pixel_t)magic_div_rt<pixel_t>((uint32_t)avg * (uint32_t)opacity_i + (uint32_t)half, magic);
  }
}

// ---------------------------------------------------------------------------
// prepare_effective_mask_for_row_sse41
// full_opacity == true  (default): MASK444 returns maskp; others fill buf with
//   spatial averages only.
// full_opacity == false: opacity baked in for every mode including MASK444.
// opacity_i, half, magic are ignored when full_opacity == true.
// ---------------------------------------------------------------------------
template<MaskMode maskMode, typename pixel_t, bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
const pixel_t* prepare_effective_mask_for_row_sse41(
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
      // Copy row with opacity scaling into buf
      pixel_t* dst = buf.data();
      int x = 0;
      if constexpr (sizeof(pixel_t) == 1) {
        const __m128i v_opacity = _mm_set1_epi16((short)opacity_i);
        const __m128i v_half16  = _mm_set1_epi16((short)half);
        const __m128i v_mdiv    = _mm_set1_epi16((short)magic.div);
        for (; x <= width - 8; x += 8) {
          __m128i v   = _mm_loadl_epi64((const __m128i*)(maskp + x));
          __m128i v16 = _mm_cvtepu8_epi16(v);
          __m128i scaled = _mm_add_epi16(_mm_mullo_epi16(v16, v_opacity), v_half16);
          __m128i res    = _mm_srli_epi16(_mm_mulhi_epu16(scaled, v_mdiv), magic.shift);
          _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi16(res, res));
        }
      } else {
        const __m128i v_opacity32 = _mm_set1_epi32(opacity_i);
        const __m128i v_half32    = _mm_set1_epi32(half);
        for (; x <= width - 4; x += 4) {
          __m128i v32 = _mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i*)(maskp + x)));
          __m128i res = simd_magic_div_32(
            _mm_add_epi32(_mm_mullo_epi32(v32, v_opacity32), v_half32),
            magic.div, magic.shift);
          _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi32(res, res));
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
      fill_mask422_sse41<pixel_t, full_opacity>(dst, maskp, width, opacity_i, half, magic);
    else if constexpr (maskMode == MASK422_MPEG2)
      fill_mask422_mpeg2_sse41<pixel_t, full_opacity>(dst, maskp, width, opacity_i, half, magic);
    else if constexpr (maskMode == MASK422_TOPLEFT)
      fill_mask422_topleft_sse41<pixel_t, full_opacity>(dst, maskp, width, opacity_i, half, magic);
    else if constexpr (maskMode == MASK420)
      fill_mask420_sse41<pixel_t, full_opacity>(dst, maskp, mask_pitch, width, opacity_i, half, magic);
    else if constexpr (maskMode == MASK420_MPEG2)
      fill_mask420_mpeg2_sse41<pixel_t, full_opacity>(dst, maskp, mask_pitch, width, opacity_i, half, magic);
    else if constexpr (maskMode == MASK420_TOPLEFT)
      fill_mask420_topleft_sse41<pixel_t, full_opacity>(dst, maskp, mask_pitch, width, opacity_i, half, magic);
    else if constexpr (maskMode == MASK411)
      fill_mask411_sse41<pixel_t, full_opacity>(dst, maskp, width, opacity_i, half, magic);
    return dst;
  }
}

// ---------------------------------------------------------------------------
// prepare_effective_mask_for_row_level_baked_sse41
// Layer-style level baking: result = (avg * level + 1) >> bits_per_pixel.
//
// Two-pass for subsampled modes:
//   Pass 1 — spatial average via fill_*<pixel_t, true> (Overlay opacity baking suppressed).
//   Pass 2 — vectorized (avg * level + 1) >> bpp over the scratch buffer.
// Single pass for MASK444.
//
// full_opacity == true  (level >= 1<<bpp): spatial average only (identity for MASK444).
// full_opacity == false: spatial average + level baking into output.
// ---------------------------------------------------------------------------
template<MaskMode maskMode, typename pixel_t, bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
const pixel_t* prepare_effective_mask_for_row_level_baked_sse41(
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
      const __m128i v_level = _mm_set1_epi16((short)level);
      const __m128i v_one   = _mm_set1_epi16(1);
      for (; x <= width - 8; x += 8) {
        __m128i v = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(maskp + x)));
        __m128i r = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(v, v_level), v_one), 8);
        _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi16(r, r));
      }
    } else {
      const __m128i v_level32 = _mm_set1_epi32(level);
      const __m128i v_one32   = _mm_set1_epi32(1);
      const __m128i v_bpp     = _mm_cvtsi32_si128(bits_per_pixel);
      for (; x <= width - 4; x += 4) {
        __m128i v = _mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i*)(maskp + x)));
        __m128i r = _mm_srl_epi32(_mm_add_epi32(_mm_mullo_epi32(v, v_level32), v_one32), v_bpp);
        _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi32(r, r));
      }
    }
    for (; x < width; x++)
      dst[x] = static_cast<pixel_t>(((uint32_t)maskp[x] * (uint32_t)level + 1u) >> bits_per_pixel);
    return dst;
  }
  else {
    pixel_t* dst = buf.data();
    // Pass 1: spatial average, Overlay baking suppressed (full_opacity=true).
    if constexpr (maskMode == MASK422)
      fill_mask422_sse41<pixel_t, true>(dst, maskp, width, 0, 0, {});
    else if constexpr (maskMode == MASK422_MPEG2)
      fill_mask422_mpeg2_sse41<pixel_t, true>(dst, maskp, width, 0, 0, {});
    else if constexpr (maskMode == MASK422_TOPLEFT)
      fill_mask422_topleft_sse41<pixel_t, true>(dst, maskp, width, 0, 0, {});
    else if constexpr (maskMode == MASK420)
      fill_mask420_sse41<pixel_t, true>(dst, maskp, mask_pitch, width, 0, 0, {});
    else if constexpr (maskMode == MASK420_MPEG2)
      fill_mask420_mpeg2_sse41<pixel_t, true>(dst, maskp, mask_pitch, width, 0, 0, {});
    else if constexpr (maskMode == MASK420_TOPLEFT)
      fill_mask420_topleft_sse41<pixel_t, true>(dst, maskp, mask_pitch, width, 0, 0, {});
    else if constexpr (maskMode == MASK411)
      fill_mask411_sse41<pixel_t, true>(dst, maskp, width, 0, 0, {});
    // Pass 2: level baking (only when !full_opacity).
    if constexpr (!full_opacity) {
      int x = 0;
      if constexpr (sizeof(pixel_t) == 1) {
        const __m128i v_level = _mm_set1_epi16((short)level);
        const __m128i v_one   = _mm_set1_epi16(1);
        for (; x <= width - 8; x += 8) {
          __m128i v = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(dst + x)));
          __m128i r = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(v, v_level), v_one), 8);
          _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi16(r, r));
        }
      } else {
        const __m128i v_level32 = _mm_set1_epi32(level);
        const __m128i v_one32   = _mm_set1_epi32(1);
        const __m128i v_bpp     = _mm_cvtsi32_si128(bits_per_pixel);
        for (; x <= width - 4; x += 4) {
          __m128i v = _mm_cvtepu16_epi32(_mm_loadl_epi64((__m128i*)(dst + x)));
          __m128i r = _mm_srl_epi32(_mm_add_epi32(_mm_mullo_epi32(v, v_level32), v_one32), v_bpp);
          _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi32(r, r));
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

// prepare_effective_mask_for_row_sse41
#define INST_PREP_SSE41(mm, pt) \
  template const pt* prepare_effective_mask_for_row_sse41<mm, pt, true> (const pt*, int, int, std::vector<pt>&, int, int, MagicDiv); \
  template const pt* prepare_effective_mask_for_row_sse41<mm, pt, false>(const pt*, int, int, std::vector<pt>&, int, int, MagicDiv);
INST_PREP_SSE41(MASK444,          uint8_t)   INST_PREP_SSE41(MASK444,          uint16_t)
INST_PREP_SSE41(MASK420,          uint8_t)   INST_PREP_SSE41(MASK420,          uint16_t)
INST_PREP_SSE41(MASK420_MPEG2,    uint8_t)   INST_PREP_SSE41(MASK420_MPEG2,    uint16_t)
INST_PREP_SSE41(MASK420_TOPLEFT,  uint8_t)   INST_PREP_SSE41(MASK420_TOPLEFT,  uint16_t)
INST_PREP_SSE41(MASK422,          uint8_t)   INST_PREP_SSE41(MASK422,          uint16_t)
INST_PREP_SSE41(MASK422_MPEG2,    uint8_t)   INST_PREP_SSE41(MASK422_MPEG2,    uint16_t)
INST_PREP_SSE41(MASK422_TOPLEFT,  uint8_t)   INST_PREP_SSE41(MASK422_TOPLEFT,  uint16_t)
INST_PREP_SSE41(MASK411,          uint8_t)   INST_PREP_SSE41(MASK411,          uint16_t)
#undef INST_PREP_SSE41

// prepare_effective_mask_for_row_level_baked_sse41
#define INST_BAKED_SSE41(mm, pt) \
  template const pt* prepare_effective_mask_for_row_level_baked_sse41<mm, pt, true> (const pt*, int, int, std::vector<pt>&, int, int); \
  template const pt* prepare_effective_mask_for_row_level_baked_sse41<mm, pt, false>(const pt*, int, int, std::vector<pt>&, int, int);
INST_BAKED_SSE41(MASK444,          uint8_t)   INST_BAKED_SSE41(MASK444,          uint16_t)
INST_BAKED_SSE41(MASK420,          uint8_t)   INST_BAKED_SSE41(MASK420,          uint16_t)
INST_BAKED_SSE41(MASK420_MPEG2,    uint8_t)   INST_BAKED_SSE41(MASK420_MPEG2,    uint16_t)
INST_BAKED_SSE41(MASK420_TOPLEFT,  uint8_t)   INST_BAKED_SSE41(MASK420_TOPLEFT,  uint16_t)
INST_BAKED_SSE41(MASK422,          uint8_t)   INST_BAKED_SSE41(MASK422,          uint16_t)
INST_BAKED_SSE41(MASK422_MPEG2,    uint8_t)   INST_BAKED_SSE41(MASK422_MPEG2,    uint16_t)
INST_BAKED_SSE41(MASK422_TOPLEFT,  uint8_t)   INST_BAKED_SSE41(MASK422_TOPLEFT,  uint16_t)
INST_BAKED_SSE41(MASK411,          uint8_t)   INST_BAKED_SSE41(MASK411,          uint16_t)
#undef INST_BAKED_SSE41
