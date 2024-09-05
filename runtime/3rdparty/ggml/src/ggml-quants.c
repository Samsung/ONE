#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include "ggml-quants.h"
#include "ggml-impl.h"


#include <math.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <stdlib.h> // for qsort
#include <stdio.h>  // for GGML_ASSERT

#define GROUP_MAX_EPS 1e-15f
#define GROUP_MAX_EPS_IQ3_XXS 1e-8f
#define GROUP_MAX_EPS_IQ2_S 1e-8f
#define GROUP_MAX_EPS_IQ1_M 1e-7f
#define GROUP_MAX_EPS_IQ1_S 1e-12f

#if defined(_MSC_VER)
// disable "possible loss of data" to avoid warnings for hundreds of casts
// we should just be careful :)
#pragma warning(disable: 4244 4267)
#endif

#define UNUSED GGML_UNUSED

// some compilers don't provide _mm256_set_m128i, e.g. gcc 7
#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)
// multiply int8_t, add results pairwise twice
static inline __m128i mul_sum_i8_pairs(const __m128i x, const __m128i y) {
    // Get absolute values of x vectors
    const __m128i ax = _mm_sign_epi8(x, x);
    // Sign the values of the y vectors
    const __m128i sy = _mm_sign_epi8(y, x);
    // Perform multiplication and create 16-bit values
    const __m128i dot = _mm_maddubs_epi16(ax, sy);
    const __m128i ones = _mm_set1_epi16(1);
    return _mm_madd_epi16(ones, dot);
}

#if __AVX__ || __AVX2__ || __AVX512F__
// horizontally add 8 floats
static inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

// horizontally add 8 int32_t
static inline int hsum_i32_8(const __m256i a) {
    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
    const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

// horizontally add 4 int32_t
static inline int hsum_i32_4(const __m128i a) {
    const __m128i hi64 = _mm_unpackhi_epi64(a, a);
    const __m128i sum64 = _mm_add_epi32(hi64, a);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

#if defined(__AVX2__) || defined(__AVX512F__)
// spread 32 bits to 32 bytes { 0x00, 0xFF }
static inline __m256i bytes_from_bits_32(const uint8_t * x) {
    uint32_t x32;
    memcpy(&x32, x, sizeof(uint32_t));
    const __m256i shuf_mask = _mm256_set_epi64x(
            0x0303030303030303, 0x0202020202020202,
            0x0101010101010101, 0x0000000000000000);
    __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(x32), shuf_mask);
    const __m256i bit_mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
    bytes = _mm256_or_si256(bytes, bit_mask);
    return _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1));
}

// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
static inline __m256i bytes_from_nibbles_32(const uint8_t * rsi)
{
    const __m128i tmp = _mm_loadu_si128((const __m128i *)rsi);
    const __m256i bytes = MM256_SET_M128I(_mm_srli_epi16(tmp, 4), tmp);
    const __m256i lowMask = _mm256_set1_epi8( 0xF );
    return _mm256_and_si256(lowMask, bytes);
}

// add int16_t pairwise and return as float vector
static inline __m256 sum_i16_pairs_float(const __m256i x) {
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i summed_pairs = _mm256_madd_epi16(ones, x);
    return _mm256_cvtepi32_ps(summed_pairs);
}

static inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
    const __m256i zero = _mm256_setzero_si256();
    const __m256i summed_pairs = _mm256_dpbusd_epi32(zero, ax, sy);
    return _mm256_cvtepi32_ps(summed_pairs);
#else
    // Perform multiplication and create 16-bit values
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    return sum_i16_pairs_float(dot);
#endif
}

// multiply int8_t, add results pairwise twice and return as float vector
static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
#if __AVXVNNIINT8__
    const __m256i zero = _mm256_setzero_si256();
    const __m256i summed_pairs = _mm256_dpbssd_epi32(zero, x, y);
    return _mm256_cvtepi32_ps(summed_pairs);
#else
    // Get absolute values of x vectors
    const __m256i ax = _mm256_sign_epi8(x, x);
    // Sign the values of the y vectors
    const __m256i sy = _mm256_sign_epi8(y, x);
    return mul_sum_us8_pairs_float(ax, sy);
#endif
}

static inline __m128i packNibbles( __m256i bytes )
{
    // Move bits within 16-bit lanes from 0000_abcd_0000_efgh into 0000_0000_abcd_efgh
#if __AVX512F__
    const __m256i bytes_srli_4 = _mm256_srli_epi16(bytes, 4);   // 0000_0000_abcd_0000
    bytes = _mm256_or_si256(bytes, bytes_srli_4);               // 0000_abcd_abcd_efgh
    return _mm256_cvtepi16_epi8(bytes);                         // abcd_efgh
#else
    const __m256i lowByte = _mm256_set1_epi16( 0xFF );
    __m256i high = _mm256_andnot_si256( lowByte, bytes );
    __m256i low = _mm256_and_si256( lowByte, bytes );
    high = _mm256_srli_epi16( high, 4 );
    bytes = _mm256_or_si256( low, high );

    // Compress uint16_t lanes into bytes
    __m128i r0 = _mm256_castsi256_si128( bytes );
    __m128i r1 = _mm256_extracti128_si256( bytes, 1 );
    return _mm_packus_epi16( r0, r1 );
#endif
}
#elif defined(__AVX__)
// spread 32 bits to 32 bytes { 0x00, 0xFF }
static inline __m256i bytes_from_bits_32(const uint8_t * x) {
    uint32_t x32;
    memcpy(&x32, x, sizeof(uint32_t));
    const __m128i shuf_maskl = _mm_set_epi64x(0x0101010101010101, 0x0000000000000000);
    const __m128i shuf_maskh = _mm_set_epi64x(0x0303030303030303, 0x0202020202020202);
    __m128i bytesl = _mm_shuffle_epi8(_mm_set1_epi32(x32), shuf_maskl);
    __m128i bytesh = _mm_shuffle_epi8(_mm_set1_epi32(x32), shuf_maskh);
    const __m128i bit_mask = _mm_set1_epi64x(0x7fbfdfeff7fbfdfe);
    bytesl = _mm_or_si128(bytesl, bit_mask);
    bytesh = _mm_or_si128(bytesh, bit_mask);
    bytesl = _mm_cmpeq_epi8(bytesl, _mm_set1_epi64x(-1));
    bytesh = _mm_cmpeq_epi8(bytesh, _mm_set1_epi64x(-1));
    return MM256_SET_M128I(bytesh, bytesl);
}

// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
static inline __m256i bytes_from_nibbles_32(const uint8_t * rsi)
{
    // Load 16 bytes from memory
    __m128i tmpl = _mm_loadu_si128((const __m128i *)rsi);
    __m128i tmph = _mm_srli_epi16(tmpl, 4);
    const __m128i lowMask = _mm_set1_epi8(0xF);
    tmpl = _mm_and_si128(lowMask, tmpl);
    tmph = _mm_and_si128(lowMask, tmph);
    return MM256_SET_M128I(tmph, tmpl);
}

// add int16_t pairwise and return as float vector
static inline __m256 sum_i16_pairs_float(const __m128i xh, const __m128i xl) {
    const __m128i ones = _mm_set1_epi16(1);
    const __m128i summed_pairsl = _mm_madd_epi16(ones, xl);
    const __m128i summed_pairsh = _mm_madd_epi16(ones, xh);
    const __m256i summed_pairs = MM256_SET_M128I(summed_pairsh, summed_pairsl);
    return _mm256_cvtepi32_ps(summed_pairs);
}

static inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
    const __m128i axl = _mm256_castsi256_si128(ax);
    const __m128i axh = _mm256_extractf128_si256(ax, 1);
    const __m128i syl = _mm256_castsi256_si128(sy);
    const __m128i syh = _mm256_extractf128_si256(sy, 1);
    // Perform multiplication and create 16-bit values
    const __m128i dotl = _mm_maddubs_epi16(axl, syl);
    const __m128i doth = _mm_maddubs_epi16(axh, syh);
    return sum_i16_pairs_float(doth, dotl);
}

// multiply int8_t, add results pairwise twice and return as float vector
static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
    const __m128i xl = _mm256_castsi256_si128(x);
    const __m128i xh = _mm256_extractf128_si256(x, 1);
    const __m128i yl = _mm256_castsi256_si128(y);
    const __m128i yh = _mm256_extractf128_si256(y, 1);
    // Get absolute values of x vectors
    const __m128i axl = _mm_sign_epi8(xl, xl);
    const __m128i axh = _mm_sign_epi8(xh, xh);
    // Sign the values of the y vectors
    const __m128i syl = _mm_sign_epi8(yl, xl);
    const __m128i syh = _mm_sign_epi8(yh, xh);
    // Perform multiplication and create 16-bit values
    const __m128i dotl = _mm_maddubs_epi16(axl, syl);
    const __m128i doth = _mm_maddubs_epi16(axh, syh);
    return sum_i16_pairs_float(doth, dotl);
}

static inline __m128i packNibbles( __m128i bytes1, __m128i bytes2 )
{
    // Move bits within 16-bit lanes from 0000_abcd_0000_efgh into 0000_0000_abcd_efgh
    const __m128i lowByte = _mm_set1_epi16( 0xFF );
    __m128i high = _mm_andnot_si128( lowByte, bytes1 );
    __m128i low = _mm_and_si128( lowByte, bytes1 );
    high = _mm_srli_epi16( high, 4 );
    bytes1 = _mm_or_si128( low, high );
    high = _mm_andnot_si128( lowByte, bytes2 );
    low = _mm_and_si128( lowByte, bytes2 );
    high = _mm_srli_epi16( high, 4 );
    bytes2 = _mm_or_si128( low, high );

    return _mm_packus_epi16( bytes1, bytes2);
}
#endif
#elif defined(__SSSE3__)
// horizontally add 4x4 floats
static inline float hsum_float_4x4(const __m128 a, const __m128 b, const __m128 c, const __m128 d) {
    __m128 res_0 =_mm_hadd_ps(a, b);
    __m128 res_1 =_mm_hadd_ps(c, d);
    __m128 res =_mm_hadd_ps(res_0, res_1);
    res =_mm_hadd_ps(res, res);
    res =_mm_hadd_ps(res, res);

    return _mm_cvtss_f32(res);
}
#endif // __AVX__ || __AVX2__ || __AVX512F__
#endif // defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)

// [Line:660]
// reference implementation for deterministic creation of model files
void quantize_row_q4_0_ref(const float * restrict x, block_q4_0 * restrict y, int64_t k) {
    static const int qk = QK4_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max  = v;
            }
        }

        const float d  = max / -8;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < qk/2; ++j) {
            const float x0 = x[i*qk + 0    + j]*id;
            const float x1 = x[i*qk + qk/2 + j]*id;

            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

            y[i].qs[j]  = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}

void quantize_row_q4_0(const float * restrict x, void * restrict y, int64_t k) {
    quantize_row_q4_0_ref(x, y, k);
}

// [Line:840]
// reference implementation for deterministic creation of model files
void quantize_row_q8_0_ref(const float * restrict x, block_q8_0 * restrict y, int64_t k) {
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK8_0; j++) {
            const float v = x[i*QK8_0 + j];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < QK8_0; ++j) {
            const float x0 = x[i*QK8_0 + j]*id;

            y[i].qs[j] = roundf(x0);
        }
    }
}

void quantize_row_q8_0(const float * restrict x, void * restrict vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0 * restrict y = vy;

#if defined(__ARM_NEON)
    for (int i = 0; i < nb; i++) {
        float32x4_t srcv [8];
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j]  = vld1q_f32(x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2*j] = vmaxq_f32(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = vmaxq_f32(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = vmaxq_f32(amaxv[8*j], amaxv[8*j+4]);

        const float amax = vmaxvq_f32(amaxv[0]);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < 8; j++) {
            const float32x4_t v  = vmulq_n_f32(srcv[j], id);
            const int32x4_t   vi = vcvtnq_s32_f32(v);

            y[i].qs[4*j + 0] = vgetq_lane_s32(vi, 0);
            y[i].qs[4*j + 1] = vgetq_lane_s32(vi, 1);
            y[i].qs[4*j + 2] = vgetq_lane_s32(vi, 2);
            y[i].qs[4*j + 3] = vgetq_lane_s32(vi, 3);
        }
    }
#elif defined(__wasm_simd128__)
    for (int i = 0; i < nb; i++) {
        v128_t srcv [8];
        v128_t asrcv[8];
        v128_t amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j]  = wasm_v128_load(x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = wasm_f32x4_abs(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2*j] = wasm_f32x4_max(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = wasm_f32x4_max(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = wasm_f32x4_max(amaxv[8*j], amaxv[8*j+4]);

        const float amax = MAX(MAX(wasm_f32x4_extract_lane(amaxv[0], 0),
                                   wasm_f32x4_extract_lane(amaxv[0], 1)),
                               MAX(wasm_f32x4_extract_lane(amaxv[0], 2),
                                   wasm_f32x4_extract_lane(amaxv[0], 3)));

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < 8; j++) {
            const v128_t v  = wasm_f32x4_mul(srcv[j], wasm_f32x4_splat(id));
            const v128_t vi = wasm_i32x4_trunc_sat_f32x4(v);

            y[i].qs[4*j + 0] = wasm_i32x4_extract_lane(vi, 0);
            y[i].qs[4*j + 1] = wasm_i32x4_extract_lane(vi, 1);
            y[i].qs[4*j + 2] = wasm_i32x4_extract_lane(vi, 2);
            y[i].qs[4*j + 3] = wasm_i32x4_extract_lane(vi, 3);
        }
    }
#elif defined(__AVX2__) || defined(__AVX__)
    for (int i = 0; i < nb; i++) {
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps( x );
        __m256 v1 = _mm256_loadu_ps( x + 8 );
        __m256 v2 = _mm256_loadu_ps( x + 16 );
        __m256 v3 = _mm256_loadu_ps( x + 24 );
        x += 32;

        // Compute max(abs(e)) for the block
        const __m256 signBit = _mm256_set1_ps( -0.0f );
        __m256 maxAbs = _mm256_andnot_ps( signBit, v0 );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v1 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v2 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v3 ) );

        __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
        max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
        max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
        const float maxScalar = _mm_cvtss_f32( max4 );

        // Quantize these floats
        const float d = maxScalar / 127.f;
        y[i].d = GGML_FP32_TO_FP16(d);
        const float id = ( maxScalar != 0.0f ) ? 127.f / maxScalar : 0.0f;
        const __m256 mul = _mm256_set1_ps( id );

        // Apply the multiplier
        v0 = _mm256_mul_ps( v0, mul );
        v1 = _mm256_mul_ps( v1, mul );
        v2 = _mm256_mul_ps( v2, mul );
        v3 = _mm256_mul_ps( v3, mul );

        // Round to nearest integer
        v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
        v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
        v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
        v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

        // Convert floats to integers
        __m256i i0 = _mm256_cvtps_epi32( v0 );
        __m256i i1 = _mm256_cvtps_epi32( v1 );
        __m256i i2 = _mm256_cvtps_epi32( v2 );
        __m256i i3 = _mm256_cvtps_epi32( v3 );

#if defined(__AVX2__)
        // Convert int32 to int16
        i0 = _mm256_packs_epi32( i0, i1 );	// 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
        i2 = _mm256_packs_epi32( i2, i3 );	// 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                            // Convert int16 to int8
        i0 = _mm256_packs_epi16( i0, i2 );	// 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

        // We got our precious signed bytes, but the order is now wrong
        // These AVX2 pack instructions process 16-byte pieces independently
        // The following instruction is fixing the order
        const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
        i0 = _mm256_permutevar8x32_epi32( i0, perm );

        _mm256_storeu_si256((__m256i *)y[i].qs, i0);
#else
        // Since we don't have in AVX some necessary functions,
        // we split the registers in half and call AVX2 analogs from SSE
        __m128i ni0 = _mm256_castsi256_si128( i0 );
        __m128i ni1 = _mm256_extractf128_si256( i0, 1);
        __m128i ni2 = _mm256_castsi256_si128( i1 );
        __m128i ni3 = _mm256_extractf128_si256( i1, 1);
        __m128i ni4 = _mm256_castsi256_si128( i2 );
        __m128i ni5 = _mm256_extractf128_si256( i2, 1);
        __m128i ni6 = _mm256_castsi256_si128( i3 );
        __m128i ni7 = _mm256_extractf128_si256( i3, 1);

        // Convert int32 to int16
        ni0 = _mm_packs_epi32( ni0, ni1 );
        ni2 = _mm_packs_epi32( ni2, ni3 );
        ni4 = _mm_packs_epi32( ni4, ni5 );
        ni6 = _mm_packs_epi32( ni6, ni7 );
        // Convert int16 to int8
        ni0 = _mm_packs_epi16( ni0, ni2 );
        ni4 = _mm_packs_epi16( ni4, ni6 );

        _mm_storeu_si128((__m128i *)(y[i].qs +  0), ni0);
        _mm_storeu_si128((__m128i *)(y[i].qs + 16), ni4);
#endif
    }
#elif defined(__riscv_v_intrinsic)

    size_t vl = __riscv_vsetvl_e32m4(QK8_0);

    for (int i = 0; i < nb; i++) {
        // load elements
        vfloat32m4_t v_x   = __riscv_vle32_v_f32m4(x+i*QK8_0, vl);

        vfloat32m4_t vfabs = __riscv_vfabs_v_f32m4(v_x, vl);
        vfloat32m1_t tmp   = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t vmax  = __riscv_vfredmax_vs_f32m4_f32m1(vfabs, tmp, vl);
        float amax = __riscv_vfmv_f_s_f32m1_f32(vmax);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        vfloat32m4_t x0 = __riscv_vfmul_vf_f32m4(v_x, id, vl);

        // convert to integer
        vint16m2_t   vi = __riscv_vfncvt_x_f_w_i16m2(x0, vl);
        vint8m1_t    vs = __riscv_vncvt_x_x_w_i8m1(vi, vl);

        // store result
        __riscv_vse8_v_i8m1(y[i].qs , vs, vl);
    }

#elif defined(__POWER9_VECTOR__)
    for (int i = 0; i < nb; i++) {
        vector float srcv [8];
        vector float asrcv[8];
        vector float amaxv[8];
        vector signed int vi[8];

        for (int j = 0; j < 8; j++) srcv[j]  = vec_xl(0, x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = vec_abs(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2*j] = vec_max(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = vec_max(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = vec_max(amaxv[8*j], amaxv[8*j+4]);

        const float amax = MAX(MAX(vec_extract(amaxv[0], 0),
                                   vec_extract(amaxv[0], 1)),
                               MAX(vec_extract(amaxv[0], 2),
                                   vec_extract(amaxv[0], 3)));

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;
        const vector float vid = vec_splats(id);

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < 8; j++) {
            const vector float v  = vec_round(vec_mul(srcv[j], vid));
            vi[j] = vec_cts(v, 0);
        }
        vec_xst(vec_pack(vec_pack(vi[0], vi[1]), vec_pack(vi[2], vi[3])),  0, &y[i].qs[0]);
        vec_xst(vec_pack(vec_pack(vi[4], vi[5]), vec_pack(vi[6], vi[7])), 16, &y[i].qs[0]);
    }

#elif defined(__loongarch_asx)
    for (int i = 0; i < nb; i++) {
        ft_union fi;
        __m256 v0 = (__m256)__lasx_xvld( x , 0);
        __m256 v1 = (__m256)__lasx_xvld( x , 32);
        __m256 v2 = (__m256)__lasx_xvld( x , 64);
        __m256 v3 = (__m256)__lasx_xvld( x , 96);
        x += 32;

        // Compute max(abs(e)) for the block
        const __m256 sign_bit = __lasx_xvreplfr2vr_s( -0.0f );
        __m256 max_abs = (__m256)__lasx_xvandn_v( (__m256i)sign_bit, (__m256i)v0 );
        max_abs = __lasx_xvfmax_s( max_abs, (__m256)__lasx_xvandn_v( (__m256i)sign_bit, (__m256i)v1 ) );
        max_abs = __lasx_xvfmax_s( max_abs, (__m256)__lasx_xvandn_v( (__m256i)sign_bit, (__m256i)v2 ) );
        max_abs = __lasx_xvfmax_s( max_abs, (__m256)__lasx_xvandn_v( (__m256i)sign_bit, (__m256i)v3 ) );

        __m128 max4 = __lsx_vfmax_s( lasx_extractf128( max_abs, 1 ), lasx_extractf128( max_abs , 0) );
        max4 = __lsx_vfmax_s( max4, (__m128)__lsx_vpickod_d((__m128i) max4, (__m128i)max4 ) );
        __m128 tmp = max4;
        max4 = __lsx_vfmax_s( max4, (__m128)__lsx_vinsgr2vr_w(tmp, __lsx_vpickve2gr_w( max4, 1 ), 0 ));
        fi.i = __lsx_vpickve2gr_w( (__m128i)max4, 0 );
        const float max_scalar = fi.f;

        // Quantize these floats
        const float d = max_scalar / 127.f;
        y[i].d = GGML_FP32_TO_FP16(d);
        const float id = ( max_scalar != 0.0f ) ? 127.f / max_scalar : 0.0f;
        const __m256 mul = (__m256)__lasx_xvreplfr2vr_s( id );

        // Apply the multiplier
        v0 = __lasx_xvfmul_s( v0, mul );
        v1 = __lasx_xvfmul_s( v1, mul );
        v2 = __lasx_xvfmul_s( v2, mul );
        v3 = __lasx_xvfmul_s( v3, mul );

        // Round to nearest integer
        __m256i i0 = __lasx_xvftintrne_w_s( v0 );
        __m256i i1 = __lasx_xvftintrne_w_s( v1 );
        __m256i i2 = __lasx_xvftintrne_w_s( v2 );
        __m256i i3 = __lasx_xvftintrne_w_s( v3 );

        __m128i ni0 = lasx_extracti128( i0, 0 );
        __m128i ni1 = lasx_extracti128( i0, 1);
        __m128i ni2 = lasx_extracti128( i1, 0);
        __m128i ni3 = lasx_extracti128( i1, 1);
        __m128i ni4 = lasx_extracti128( i2, 0);
        __m128i ni5 = lasx_extracti128( i2, 1);
        __m128i ni6 = lasx_extracti128( i3, 0);
        __m128i ni7 = lasx_extracti128( i3, 1);

        // Convert int32 to int16
        ni0 = lsx_packs_w( ni0, ni1 );
        ni2 = lsx_packs_w( ni2, ni3 );
        ni4 = lsx_packs_w( ni4, ni5 );
        ni6 = lsx_packs_w( ni6, ni7 );
        // Convert int16 to int8
        ni0 = lsx_packs_h( ni0, ni2 );
        ni4 = lsx_packs_h( ni4, ni6 );

        __lsx_vst(ni0, (__m128i *)(y[i].qs +  0), 0);
        __lsx_vst(ni4, (__m128i *)(y[i].qs + 16), 0);

    }
#else
    GGML_UNUSED(nb);
    // scalar
    quantize_row_q8_0_ref(x, y, k);
#endif
}

// [Line:1515]
void dequantize_row_q4_0(const block_q4_0 * restrict x, float * restrict y, int64_t k) {
    static const int qk = QK4_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >>   4) - 8;

            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}

// [Line:1609]
void dequantize_row_q8_0(const block_q8_0 * restrict x, float * restrict y, int64_t k) {
    static const int qk = QK8_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int j = 0; j < qk; ++j) {
            y[i*qk + j] = x[i].qs[j]*d;
        }
    }
}

// [Line:3729]
void ggml_vec_dot_q4_0_q8_0(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);
#if defined(__ARM_FEATURE_MATMUL_INT8)
    assert((nrc == 2) || (nrc == 1));
#else
    assert(nrc == 1);
#endif
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_0 * restrict x = vx;
    const block_q8_0 * restrict y = vy;

#if defined(__ARM_FEATURE_MATMUL_INT8)
    if (nrc == 2) {
        const block_q4_0 * restrict vx0 = vx;
        const block_q4_0 * restrict vx1 = (const block_q4_0 *) ((const uint8_t*)vx + bx);
        const block_q8_0 * restrict vy0 = vy;
        const block_q8_0 * restrict vy1 = (const block_q8_0 *) ((const uint8_t*)vy + by);

        float32x4_t sumv0 = vdupq_n_f32(0.0f);

        for (int i = 0; i < nb; i++) {
            const block_q4_0 * restrict b_x0 = &vx0[i];
            const block_q4_0 * restrict b_x1 = &vx1[i];
            const block_q8_0 * restrict b_y0 = &vy0[i];
            const block_q8_0 * restrict b_y1 = &vy1[i];

            const uint8x16_t m4b = vdupq_n_u8(0x0F);
            const int8x16_t  s8b = vdupq_n_s8(0x8);

            const uint8x16_t v0_0 = vld1q_u8(b_x0->qs);
            const uint8x16_t v0_1 = vld1q_u8(b_x1->qs);

            // 4-bit -> 8-bit
            const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
            const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
            const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
            const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

            // sub 8
            const int8x16_t x0_l = vsubq_s8(v0_0l, s8b);
            const int8x16_t x0_h = vsubq_s8(v0_0h, s8b);
            const int8x16_t x1_l = vsubq_s8(v0_1l, s8b);
            const int8x16_t x1_h = vsubq_s8(v0_1h, s8b);

            // load y
            const int8x16_t y0_l = vld1q_s8(b_y0->qs);
            const int8x16_t y0_h = vld1q_s8(b_y0->qs + 16);
            const int8x16_t y1_l = vld1q_s8(b_y1->qs);
            const int8x16_t y1_h = vld1q_s8(b_y1->qs + 16);

            float32_t _scale[4] = { GGML_FP16_TO_FP32(b_x0->d)*GGML_FP16_TO_FP32(b_y0->d),
                                    GGML_FP16_TO_FP32(b_x0->d)*GGML_FP16_TO_FP32(b_y1->d),
                                    GGML_FP16_TO_FP32(b_x1->d)*GGML_FP16_TO_FP32(b_y0->d),
                                    GGML_FP16_TO_FP32(b_x1->d)*GGML_FP16_TO_FP32(b_y1->d)};

            float32x4_t scale = vld1q_f32(_scale);

            int8x16_t l0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));
            int8x16_t l1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));

            int8x16_t l2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));
            int8x16_t l3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));

            int8x16_t r0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));
            int8x16_t r1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));

            int8x16_t r2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));
            int8x16_t r3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));

            sumv0 = vmlaq_f32(sumv0,(vcvtq_f32_s32(vmmlaq_s32((vmmlaq_s32((vmmlaq_s32((vmmlaq_s32(vdupq_n_s32(0), l0, r0)),
                                                                                l1, r1)), l2, r2)), l3, r3))), scale);
        }
        float32x4_t sumv1 = vextq_f32(sumv0, sumv0, 2);
        float32x4_t sumv2 = vzip1q_f32(sumv0, sumv1);

        vst1_f32(s,      vget_low_f32(sumv2));
        vst1_f32(s + bs, vget_high_f32(sumv2));
        return;
    }
#endif

    int ib = 0;
    float sumf = 0;

#if defined(__ARM_FEATURE_SVE)
    if (ggml_sve_cnt_b == QK8_0) {
        const svbool_t ptrueh = svptrue_pat_b8(SV_VL16);
        const svbool_t ptruel = svnot_b_z(svptrue_b8(), ptrueh);

        svfloat32_t sumv0 = svdup_n_f32(0.0f);
        svfloat32_t sumv1 = svdup_n_f32(0.0f);

        for (; ib + 1 < nb; ib += 2) {
            const block_q4_0 * restrict x0 = &x[ib + 0];
            const block_q4_0 * restrict x1 = &x[ib + 1];
            const block_q8_0 * restrict y0 = &y[ib + 0];
            const block_q8_0 * restrict y1 = &y[ib + 1];

            // load x
            const svuint8_t qx0r = svld1rq_u8(svptrue_b8(), x0->qs);
            const svuint8_t qx1r = svld1rq_u8(svptrue_b8(), x1->qs);

            // 4-bit -> 8-bit
            const svint8_t qx0 = svreinterpret_s8_u8(svlsr_n_u8_m(ptruel, svand_n_u8_m(ptrueh, qx0r, 0x0F), 0x04));
            const svint8_t qx1 = svreinterpret_s8_u8(svlsr_n_u8_m(ptruel, svand_n_u8_m(ptrueh, qx1r, 0x0F), 0x04));

            // sub 8
            const svint8_t qx0s = svsub_n_s8_x(svptrue_b8(), qx0, 8);
            const svint8_t qx1s = svsub_n_s8_x(svptrue_b8(), qx1, 8);

            // load y
            const svint8_t qy0 = svld1_s8(svptrue_b8(), y0->qs);
            const svint8_t qy1 = svld1_s8(svptrue_b8(), y1->qs);

            // dot product
            sumv0 = svmla_n_f32_x(svptrue_b32(), sumv0, svcvt_f32_s32_x(svptrue_b32(), svdot_s32(svdup_n_s32(0), qx0s, qy0)), GGML_FP16_TO_FP32(x0->d)*GGML_FP16_TO_FP32(y0->d));
            sumv1 = svmla_n_f32_x(svptrue_b32(), sumv1, svcvt_f32_s32_x(svptrue_b32(), svdot_s32(svdup_n_s32(0), qx1s, qy1)), GGML_FP16_TO_FP32(x1->d)*GGML_FP16_TO_FP32(y1->d));
        }

        sumf = svaddv_f32(svptrue_b32(), svadd_f32_x(svptrue_b32(), sumv0, sumv1));
    }
#elif defined(__ARM_NEON)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    for (; ib + 1 < nb; ib += 2) {
        const block_q4_0 * restrict x0 = &x[ib + 0];
        const block_q4_0 * restrict x1 = &x[ib + 1];
        const block_q8_0 * restrict y0 = &y[ib + 0];
        const block_q8_0 * restrict y1 = &y[ib + 1];

        const uint8x16_t m4b = vdupq_n_u8(0x0F);
        const int8x16_t  s8b = vdupq_n_s8(0x8);

        const uint8x16_t v0_0 = vld1q_u8(x0->qs);
        const uint8x16_t v0_1 = vld1q_u8(x1->qs);

        // 4-bit -> 8-bit
        const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
        const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
        const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // sub 8
        const int8x16_t v0_0ls = vsubq_s8(v0_0l, s8b);
        const int8x16_t v0_0hs = vsubq_s8(v0_0h, s8b);
        const int8x16_t v0_1ls = vsubq_s8(v0_1l, s8b);
        const int8x16_t v0_1hs = vsubq_s8(v0_1h, s8b);

        // load y
        const int8x16_t v1_0l = vld1q_s8(y0->qs);
        const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
        const int8x16_t v1_1l = vld1q_s8(y1->qs);
        const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

        // dot product into int32x4_t
        const int32x4_t p_0 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), v0_0ls, v1_0l), v0_0hs, v1_0h);
        const int32x4_t p_1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), v0_1ls, v1_1l), v0_1hs, v1_1h);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), GGML_FP16_TO_FP32(x0->d)*GGML_FP16_TO_FP32(y0->d));
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), GGML_FP16_TO_FP32(x1->d)*GGML_FP16_TO_FP32(y1->d));
    }

    sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
#elif defined(__AVX2__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (; ib < nb; ++ib) {
        /* Compute combined scale for the block */
        const __m256 d = _mm256_set1_ps( GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d) );

        __m256i qx = bytes_from_nibbles_32(x[ib].qs);

        // Now we have a vector with bytes in [ 0 .. 15 ] interval. Offset them into [ -8 .. +7 ] interval.
        const __m256i off = _mm256_set1_epi8( 8 );
        qx = _mm256_sub_epi8( qx, off );

        __m256i qy = _mm256_loadu_si256((const __m256i *)y[ib].qs);

        const __m256 q = mul_sum_i8_pairs_float(qx, qy);

        /* Multiply q with scale and accumulate */
        acc = _mm256_fmadd_ps( d, q, acc );
    }

    sumf = hsum_float_8(acc);
#elif defined(__AVX__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (; ib < nb; ++ib) {
        // Compute combined scale for the block
        const __m256 d = _mm256_set1_ps( GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d) );

        const __m128i lowMask = _mm_set1_epi8(0xF);
        const __m128i off = _mm_set1_epi8(8);

        const __m128i tmp = _mm_loadu_si128((const __m128i *)x[ib].qs);

        __m128i bx_0 = _mm_and_si128(lowMask, tmp);
        __m128i by_0 = _mm_loadu_si128((const __m128i *)y[ib].qs);
        bx_0 = _mm_sub_epi8(bx_0, off);
        const __m128i i32_0 = mul_sum_i8_pairs(bx_0, by_0);

        bx_0 = _mm_and_si128(lowMask, _mm_srli_epi64(tmp, 4));
        by_0 = _mm_loadu_si128((const __m128i *)(y[ib].qs + 16));
        bx_0 = _mm_sub_epi8(bx_0, off);
        const __m128i i32_1 = mul_sum_i8_pairs(bx_0, by_0);

        // Convert int32_t to float
        __m256 p = _mm256_cvtepi32_ps(MM256_SET_M128I(i32_0, i32_1));

        // Apply the scale, and accumulate
        acc = _mm256_add_ps(_mm256_mul_ps( d, p ), acc);
    }

    sumf = hsum_float_8(acc);
#elif defined(__SSSE3__)
    // set constants
    const __m128i lowMask = _mm_set1_epi8(0xF);
    const __m128i off = _mm_set1_epi8(8);

    // Initialize accumulator with zeros
    __m128 acc_0 = _mm_setzero_ps();
    __m128 acc_1 = _mm_setzero_ps();
    __m128 acc_2 = _mm_setzero_ps();
    __m128 acc_3 = _mm_setzero_ps();

    for (; ib + 1 < nb; ib += 2) {
        _mm_prefetch(&x[ib] + sizeof(block_q4_0), _MM_HINT_T0);
        _mm_prefetch(&y[ib] + sizeof(block_q8_0), _MM_HINT_T0);

        // Compute combined scale for the block 0 and 1
        const __m128 d_0_1 = _mm_set1_ps( GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d) );

        const __m128i tmp_0_1 = _mm_loadu_si128((const __m128i *)x[ib].qs);

        __m128i bx_0 = _mm_and_si128(lowMask, tmp_0_1);
        __m128i by_0 = _mm_loadu_si128((const __m128i *)y[ib].qs);
        bx_0 = _mm_sub_epi8(bx_0, off);
        const __m128i i32_0 = mul_sum_i8_pairs(bx_0, by_0);

        __m128i bx_1 = _mm_and_si128(lowMask, _mm_srli_epi64(tmp_0_1, 4));
        __m128i by_1 = _mm_loadu_si128((const __m128i *)(y[ib].qs + 16));
        bx_1 = _mm_sub_epi8(bx_1, off);
        const __m128i i32_1 = mul_sum_i8_pairs(bx_1, by_1);

        _mm_prefetch(&x[ib] + 2 * sizeof(block_q4_0), _MM_HINT_T0);
        _mm_prefetch(&y[ib] + 2 * sizeof(block_q8_0), _MM_HINT_T0);

        // Compute combined scale for the block 2 and 3
        const __m128 d_2_3 = _mm_set1_ps( GGML_FP16_TO_FP32(x[ib + 1].d) * GGML_FP16_TO_FP32(y[ib + 1].d) );

        const __m128i tmp_2_3 = _mm_loadu_si128((const __m128i *)x[ib + 1].qs);

        __m128i bx_2 = _mm_and_si128(lowMask, tmp_2_3);
        __m128i by_2 = _mm_loadu_si128((const __m128i *)y[ib + 1].qs);
        bx_2 = _mm_sub_epi8(bx_2, off);
        const __m128i i32_2 = mul_sum_i8_pairs(bx_2, by_2);

        __m128i bx_3 = _mm_and_si128(lowMask, _mm_srli_epi64(tmp_2_3, 4));
        __m128i by_3 = _mm_loadu_si128((const __m128i *)(y[ib + 1].qs + 16));
        bx_3 = _mm_sub_epi8(bx_3, off);
        const __m128i i32_3 = mul_sum_i8_pairs(bx_3, by_3);

        // Convert int32_t to float
        __m128 p0 = _mm_cvtepi32_ps(i32_0);
        __m128 p1 = _mm_cvtepi32_ps(i32_1);
        __m128 p2 = _mm_cvtepi32_ps(i32_2);
        __m128 p3 = _mm_cvtepi32_ps(i32_3);

        // Apply the scale
        __m128 p0_d = _mm_mul_ps( d_0_1, p0 );
        __m128 p1_d = _mm_mul_ps( d_0_1, p1 );
        __m128 p2_d = _mm_mul_ps( d_2_3, p2 );
        __m128 p3_d = _mm_mul_ps( d_2_3, p3 );

        // Acummulate
        acc_0 = _mm_add_ps(p0_d, acc_0);
        acc_1 = _mm_add_ps(p1_d, acc_1);
        acc_2 = _mm_add_ps(p2_d, acc_2);
        acc_3 = _mm_add_ps(p3_d, acc_3);
    }

    sumf = hsum_float_4x4(acc_0, acc_1, acc_2, acc_3);
#elif defined(__riscv_v_intrinsic)
    size_t vl = __riscv_vsetvl_e8m1(qk/2);

    for (; ib < nb; ++ib) {
        // load elements
        vuint8mf2_t tx = __riscv_vle8_v_u8mf2(x[ib].qs, vl);

        vint8mf2_t y0 = __riscv_vle8_v_i8mf2(y[ib].qs, vl);
        vint8mf2_t y1 = __riscv_vle8_v_i8mf2(y[ib].qs+16, vl);

        // mask and store lower part of x, and then upper part
        vuint8mf2_t x_a = __riscv_vand_vx_u8mf2(tx, 0x0F, vl);
        vuint8mf2_t x_l = __riscv_vsrl_vx_u8mf2(tx, 0x04, vl);

        vint8mf2_t x_ai = __riscv_vreinterpret_v_u8mf2_i8mf2(x_a);
        vint8mf2_t x_li = __riscv_vreinterpret_v_u8mf2_i8mf2(x_l);

        // subtract offset
        vint8mf2_t v0 = __riscv_vsub_vx_i8mf2(x_ai, 8, vl);
        vint8mf2_t v1 = __riscv_vsub_vx_i8mf2(x_li, 8, vl);

        vint16m1_t vec_mul1 = __riscv_vwmul_vv_i16m1(v0, y0, vl);
        vint16m1_t vec_mul2 = __riscv_vwmul_vv_i16m1(v1, y1, vl);

        vint32m1_t vec_zero = __riscv_vmv_v_x_i32m1(0, vl);

        vint32m1_t vs1 = __riscv_vwredsum_vs_i16m1_i32m1(vec_mul1, vec_zero, vl);
        vint32m1_t vs2 = __riscv_vwredsum_vs_i16m1_i32m1(vec_mul2, vs1, vl);

        int sumi = __riscv_vmv_x_s_i32m1_i32(vs2);

        sumf += sumi*GGML_FP16_TO_FP32(x[ib].d)*GGML_FP16_TO_FP32(y[ib].d);
    }

#elif defined(__POWER9_VECTOR__)
    const vector signed char lowMask = vec_splats((signed char)0xF);
    const vector signed int v0 = vec_splats((int32_t)0);
    const vector unsigned char v4 = vec_splats((unsigned char)0x4);
    const vector signed char v8 = vec_splats((signed char)0x8);

    vector float vsumf0 = vec_splats(0.0f);

#pragma GCC unroll 8
    for (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        vector float vxd = vec_splats(GGML_FP16_TO_FP32(x[ib].d));
        vector float vyd = vec_splats(GGML_FP16_TO_FP32(y[ib].d));
        vector float vd = vec_mul(vxd, vyd);

        vector signed char qxs = (vector signed char)vec_xl( 0, x[ib].qs);
        vector signed char q8y0 = vec_xl( 0, y[ib].qs);
        vector signed char q8y1 = vec_xl(16, y[ib].qs);

        vector signed char q4x0 = vec_and(qxs, lowMask);
        vector signed char q4x1 = vec_sr(qxs, v4);

        q4x0 = vec_sub(q4x0, v8);
        q4x1 = vec_sub(q4x1, v8);

        vector signed short qv0 = vec_add(vec_mule(q4x0, q8y0), vec_mulo(q4x0, q8y0));
        vector signed short qv1 = vec_add(vec_mule(q4x1, q8y1), vec_mulo(q4x1, q8y1));

        vector signed int vsumi0 = v0;

        vsumi0 = vec_sum4s(qv0, vsumi0);
        vsumi0 = vec_sum4s(qv1, vsumi0);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_extract(vsumf0, 0);

#elif defined(__loongarch_asx)
    // Initialize accumulator with zeros
    __m256 acc = (__m256)__lasx_xvldi(0);

    // Main loop
    for (; ib < nb; ++ib) {
        /* Compute combined scale for the block */
        const __m256 d = __lasx_xvreplfr2vr_s( GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d) );

        __m256i qx = bytes_from_nibbles_32(x[ib].qs);

        // Now we have a vector with bytes in [ 0 .. 15 ] interval. Offset them into [ -8 .. +7 ] interval.
        const __m256i off = __lasx_xvreplgr2vr_b( 8 );
        qx = __lasx_xvsub_b( qx, off );

        __m256i qy = __lasx_xvld((const __m256i *)y[ib].qs, 0);

        const __m256 q = mul_sum_i8_pairs_float(qx, qy);

        /* Multiply q with scale and accumulate */
        acc = __lasx_xvfmadd_s( d, q, acc );
    }

    sumf = hsum_float_8(acc);
#elif defined(__loongarch_sx)
    // set constants
    const __m128i low_mask = __lsx_vreplgr2vr_b(0xF);
    const __m128i off = __lsx_vreplgr2vr_b(8);

    // Initialize accumulator with zeros
    __m128 acc_0 = __lsx_vldi(0);
    __m128 acc_1 = __lsx_vldi(0);
    __m128 acc_2 = __lsx_vldi(0);
    __m128 acc_3 = __lsx_vldi(0);

    for (; ib + 1 < nb; ib += 2) {

        // Compute combined scale for the block 0 and 1
        const __m128 d_0_1 = __lsx_vreplgr2vr_w( GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d) );

        const __m128i tmp_0_1 = __lsx_vld((const __m128i *)x[ib].qs, 0);

        __m128i bx_0 = __lsx_vand_v(low_mask, tmp_0_1);
        __m128i by_0 = __lsx_vld((const __m128i *)y[ib].qs, 0);
        bx_0 = __lsx_vsub_b(bx_0, off);
        const __m128i i32_0 = mul_sum_i8_pairs(bx_0, by_0);

        __m128i bx_1 = __lsx_vand_v(low_mask, __lsx_vsrli_d(tmp_0_1, 4));
        __m128i by_1 = __lsx_vld((const __m128i *)(y[ib].qs + 16), 0);
        bx_1 = __lsx_vsub_b(bx_1, off);
        const __m128i i32_1 = mul_sum_i8_pairs(bx_1, by_1);

        //_mm_prefetch(&x[ib] + 2 * sizeof(block_q4_0), _MM_HINT_T0);
        //_mm_prefetch(&y[ib] + 2 * sizeof(block_q8_0), _MM_HINT_T0);

        // Compute combined scale for the block 2 and 3
        const __m128 d_2_3 = __lsx_vreplgr2vr_w( GGML_FP16_TO_FP32(x[ib + 1].d) * GGML_FP16_TO_FP32(y[ib + 1].d) );

        const __m128i tmp_2_3 = __lsx_vld((const __m128i *)x[ib + 1].qs, 0);

        __m128i bx_2 = __lsx_vand_v(low_mask, tmp_2_3);
        __m128i by_2 = __lsx_vld((const __m128i *)y[ib + 1].qs, 0);
        bx_2 = __lsx_vsub_b(bx_2, off);
        const __m128i i32_2 = mul_sum_i8_pairs(bx_2, by_2);

        __m128i bx_3 = __lsx_vand_v(low_mask, __lsx_vsrli_d(tmp_2_3, 4));
        __m128i by_3 = __lsx_vld((const __m128i *)(y[ib + 1].qs + 16), 0);
        bx_3 = __lsx_vsub_b(bx_3, off);
        const __m128i i32_3 = mul_sum_i8_pairs(bx_3, by_3);

        // Convert int32_t to float
        __m128 p0 = __lsx_vffint_s_w(i32_0);
        __m128 p1 = __lsx_vffint_s_w(i32_1);
        __m128 p2 = __lsx_vffint_s_w(i32_2);
        __m128 p3 = __lsx_vffint_s_w(i32_3);

        // Apply the scale
        __m128 p0_d = __lsx_vfmul_s( d_0_1, p0 );
        __m128 p1_d = __lsx_vfmul_s( d_0_1, p1 );
        __m128 p2_d = __lsx_vfmul_s( d_2_3, p2 );
        __m128 p3_d = __lsx_vfmul_s( d_2_3, p3 );

        // Acummulate
        acc_0 = __lsx_vfadd_s(p0_d, acc_0);
        acc_1 = __lsx_vfadd_s(p1_d, acc_1);
        acc_2 = __lsx_vfadd_s(p2_d, acc_2);
        acc_3 = __lsx_vfadd_s(p3_d, acc_3);
    }

    sumf = hsum_float_4x4(acc_0, acc_1, acc_2, acc_3);
#endif
    for (; ib < nb; ++ib) {
        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const int v0 = (x[ib].qs[j] & 0x0F) - 8;
            const int v1 = (x[ib].qs[j] >>   4) - 8;

            sumi0 += (v0 * y[ib].qs[j]);
            sumi1 += (v1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += sumi*GGML_FP16_TO_FP32(x[ib].d)*GGML_FP16_TO_FP32(y[ib].d);
    }

    *s = sumf;
}

// [Line:5227]
void ggml_vec_dot_q8_0_q8_0(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);
#if defined(__ARM_FEATURE_MATMUL_INT8)
    assert((nrc == 2) || (nrc == 1));
#else
    assert(nrc == 1);
#endif
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q8_0 * restrict x = vx;
    const block_q8_0 * restrict y = vy;

#if defined(__ARM_FEATURE_MATMUL_INT8)
    if (nrc == 2) {
        const block_q8_0 * restrict vx0 = vx;
        const block_q8_0 * restrict vx1 = (const block_q8_0 *) ((const uint8_t*)vx + bx);
        const block_q8_0 * restrict vy0 = vy;
        const block_q8_0 * restrict vy1 = (const block_q8_0 *) ((const uint8_t*)vy + by);

        float32x4_t sumv0 = vdupq_n_f32(0.0f);

        for (int i = 0; i < nb; i++) {
            const block_q8_0 * restrict b_x0 = &vx0[i];
            const block_q8_0 * restrict b_y0 = &vy0[i];

            const block_q8_0 * restrict b_x1 = &vx1[i];
            const block_q8_0 * restrict b_y1 = &vy1[i];

            const int8x16_t x0_l = vld1q_s8(b_x0->qs);
            const int8x16_t x0_h = vld1q_s8(b_x0->qs + 16);
            const int8x16_t x1_l = vld1q_s8(b_x1->qs);
            const int8x16_t x1_h = vld1q_s8(b_x1->qs + 16);

            // load y
            const int8x16_t y0_l = vld1q_s8(b_y0->qs);
            const int8x16_t y0_h = vld1q_s8(b_y0->qs + 16);
            const int8x16_t y1_l = vld1q_s8(b_y1->qs);
            const int8x16_t y1_h = vld1q_s8(b_y1->qs + 16);

            float32_t _scale[4] = {GGML_FP16_TO_FP32(b_x0->d)*GGML_FP16_TO_FP32(b_y0->d),
                                   GGML_FP16_TO_FP32(b_x0->d)*GGML_FP16_TO_FP32(b_y1->d),
                                   GGML_FP16_TO_FP32(b_x1->d)*GGML_FP16_TO_FP32(b_y0->d),
                                   GGML_FP16_TO_FP32(b_x1->d)*GGML_FP16_TO_FP32(b_y1->d)};
            float32x4_t scale = vld1q_f32(_scale);

            int8x16_t l0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));
            int8x16_t l1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));

            int8x16_t l2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));
            int8x16_t l3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));

            int8x16_t r0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));
            int8x16_t r1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));

            int8x16_t r2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));
            int8x16_t r3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));

            sumv0 = vmlaq_f32(sumv0,(vcvtq_f32_s32(vmmlaq_s32((vmmlaq_s32((vmmlaq_s32((vmmlaq_s32(vdupq_n_s32(0), l0, r0)),
                                                                                       l1, r1)), l2, r2)), l3, r3))), scale);
        }
        float32x4_t sumv1 = vextq_f32(sumv0, sumv0, 2);
        float32x4_t sumv2 = vzip1q_f32(sumv0, sumv1);

        vst1_f32(s, vget_low_f32(sumv2));
        vst1_f32(s + bs, vget_high_f32(sumv2));
        return;
    }
#endif

    int ib = 0;
    float sumf = 0;

#if defined(__ARM_FEATURE_SVE)
    if (ggml_sve_cnt_b == QK8_0) {
        svfloat32_t sumv0 = svdup_n_f32(0.0f);
        svfloat32_t sumv1 = svdup_n_f32(0.0f);

        for (; ib + 1 < nb; ib += 2) {
            const block_q8_0 * restrict x0 = &x[ib + 0];
            const block_q8_0 * restrict x1 = &x[ib + 1];
            const block_q8_0 * restrict y0 = &y[ib + 0];
            const block_q8_0 * restrict y1 = &y[ib + 1];

            // load x
            const svint8_t qx0 = svld1_s8(svptrue_b8(), x0->qs);
            const svint8_t qx1 = svld1_s8(svptrue_b8(), x1->qs);

            // load y
            const svint8_t qy0 = svld1_s8(svptrue_b8(), y0->qs);
            const svint8_t qy1 = svld1_s8(svptrue_b8(), y1->qs);

            sumv0 = svmla_n_f32_x(svptrue_b32(), sumv0, svcvt_f32_s32_x(svptrue_b32(), svdot_s32(svdup_n_s32(0), qx0, qy0)), GGML_FP16_TO_FP32(x0->d)*GGML_FP16_TO_FP32(y0->d));
            sumv1 = svmla_n_f32_x(svptrue_b32(), sumv1, svcvt_f32_s32_x(svptrue_b32(), svdot_s32(svdup_n_s32(0), qx1, qy1)), GGML_FP16_TO_FP32(x1->d)*GGML_FP16_TO_FP32(y1->d));
        }

        sumf = svaddv_f32(svptrue_b32(), svadd_f32_x(svptrue_b32(), sumv0, sumv1));
    }
#elif defined(__ARM_NEON)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    for (; ib + 1 < nb; ib += 2) {
        const block_q8_0 * restrict x0 = &x[ib + 0];
        const block_q8_0 * restrict x1 = &x[ib + 1];
        const block_q8_0 * restrict y0 = &y[ib + 0];
        const block_q8_0 * restrict y1 = &y[ib + 1];

        const int8x16_t x0_0 = vld1q_s8(x0->qs);
        const int8x16_t x0_1 = vld1q_s8(x0->qs + 16);
        const int8x16_t x1_0 = vld1q_s8(x1->qs);
        const int8x16_t x1_1 = vld1q_s8(x1->qs + 16);

        // load y
        const int8x16_t y0_0 = vld1q_s8(y0->qs);
        const int8x16_t y0_1 = vld1q_s8(y0->qs + 16);
        const int8x16_t y1_0 = vld1q_s8(y1->qs);
        const int8x16_t y1_1 = vld1q_s8(y1->qs + 16);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(
                        ggml_vdotq_s32(vdupq_n_s32(0), x0_0, y0_0),
                        ggml_vdotq_s32(vdupq_n_s32(0), x0_1, y0_1))), GGML_FP16_TO_FP32(x0->d)*GGML_FP16_TO_FP32(y0->d));

        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(
                        ggml_vdotq_s32(vdupq_n_s32(0), x1_0, y1_0),
                        ggml_vdotq_s32(vdupq_n_s32(0), x1_1, y1_1))), GGML_FP16_TO_FP32(x1->d)*GGML_FP16_TO_FP32(y1->d));
    }

    sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
#elif defined(__AVX2__) || defined(__AVX__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (; ib < nb; ++ib) {
        // Compute combined scale for the block
        const __m256 d = _mm256_set1_ps(GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d));
        __m256i qx = _mm256_loadu_si256((const __m256i *)x[ib].qs);
        __m256i qy = _mm256_loadu_si256((const __m256i *)y[ib].qs);

        const __m256 q = mul_sum_i8_pairs_float(qx, qy);

        // Multiply q with scale and accumulate
#if defined(__AVX2__)
        acc = _mm256_fmadd_ps( d, q, acc );
#else
        acc = _mm256_add_ps( _mm256_mul_ps( d, q ), acc );
#endif
    }

    sumf = hsum_float_8(acc);
#elif defined(__riscv_v_intrinsic)
    size_t vl = __riscv_vsetvl_e8m1(qk);

    for (; ib < nb; ++ib) {
        // load elements
        vint8m1_t bx_0 = __riscv_vle8_v_i8m1(x[ib].qs, vl);
        vint8m1_t by_0 = __riscv_vle8_v_i8m1(y[ib].qs, vl);

        vint16m2_t vw_mul = __riscv_vwmul_vv_i16m2(bx_0, by_0, vl);

        vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, vl);
        vint32m1_t v_sum = __riscv_vwredsum_vs_i16m2_i32m1(vw_mul, v_zero, vl);

        int sumi = __riscv_vmv_x_s_i32m1_i32(v_sum);

        sumf += sumi*(GGML_FP16_TO_FP32(x[ib].d)*GGML_FP16_TO_FP32(y[ib].d));
    }
#elif defined(__POWER9_VECTOR__)
    const vector signed int v0 = vec_splats((int32_t)0);
    vector float vsumf0 = vec_splats(0.0f);

#pragma GCC unroll 8
    for (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        vector float vxd = vec_splats(GGML_FP16_TO_FP32(x[ib].d));
        vector float vyd = vec_splats(GGML_FP16_TO_FP32(y[ib].d));
        vector float vd = vec_mul(vxd, vyd);

        vector signed char q8x0 = vec_xl( 0, x[ib].qs);
        vector signed char q8x1 = vec_xl(16, x[ib].qs);
        vector signed char q8y0 = vec_xl( 0, y[ib].qs);
        vector signed char q8y1 = vec_xl(16, y[ib].qs);

        vector signed short qv0 = vec_mule(q8x0, q8y0);
        vector signed short qv1 = vec_mulo(q8x0, q8y0);
        vector signed short qv2 = vec_mule(q8x1, q8y1);
        vector signed short qv3 = vec_mulo(q8x1, q8y1);

        vector signed int vsumi0 = v0;
        vector signed int vsumi1 = v0;

        vsumi0 = vec_sum4s(qv0, vsumi0);
        vsumi1 = vec_sum4s(qv1, vsumi1);
        vsumi0 = vec_sum4s(qv2, vsumi0);
        vsumi1 = vec_sum4s(qv3, vsumi1);

        vsumi0 = vec_add(vsumi0, vsumi1);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_extract(vsumf0, 0);

#elif defined(__loongarch_asx)
    // Initialize accumulator with zeros
    __m256 acc = (__m256)__lasx_xvldi(0);

    // Main loop
    for (; ib < nb; ++ib) {
        // Compute combined scale for the block
        const __m256 d = __lasx_xvreplfr2vr_s(GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d));
        __m256i qx = __lasx_xvld((const __m256i *)x[ib].qs, 0);
        __m256i qy = __lasx_xvld((const __m256i *)y[ib].qs, 0);

        const __m256 q = mul_sum_i8_pairs_float(qx, qy);

        // Multiply q with scale and accumulate
        acc = __lasx_xvfmadd_s( d, q, acc );
    }

    sumf = hsum_float_8(acc);
#endif
    for (; ib < nb; ++ib) {
        int sumi = 0;

        for (int j = 0; j < qk; j++) {
            sumi += x[ib].qs[j]*y[ib].qs[j];
        }

        sumf += sumi*(GGML_FP16_TO_FP32(x[ib].d)*GGML_FP16_TO_FP32(y[ib].d));
    }

    *s = sumf;
}
